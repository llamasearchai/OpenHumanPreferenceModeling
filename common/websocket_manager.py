"""
WebSocket Connection Manager

Purpose: Manages WebSocket connections with room-based messaging,
heartbeat monitoring, and message broadcasting capabilities.

Supports:
- Per-user rooms for targeted messaging
- Broadcast to all connected clients
- Heartbeat ping/pong for connection health
- Sequence tracking for message ordering
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    METRIC_UPDATE = "metric_update"
    ALERT_UPDATE = "alert_update"
    TASK_ASSIGNED = "task_assigned"
    CALIBRATION_STATUS = "calibration_status"
    TRAINING_PROGRESS = "training_progress"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    ERROR = "error"


class WebSocketMessage(BaseModel):
    type: MessageType
    payload: Dict[str, Any]
    timestamp: str
    sequence: int


@dataclass
class ConnectionInfo:
    websocket: WebSocket
    user_id: str
    connected_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    last_sequence: int = 0
    rooms: Set[str] = field(default_factory=set)


class WebSocketManager:
    """
    Manages WebSocket connections with room-based messaging.

    Features:
    - Room-based connection grouping (by user_id, role, etc.)
    - Heartbeat monitoring with configurable interval
    - Message sequencing for gap detection
    - Automatic cleanup of stale connections
    """

    def __init__(self, heartbeat_interval: int = 30):
        # Room -> Set of connection IDs
        self._rooms: Dict[str, Set[str]] = {}
        # Connection ID -> ConnectionInfo
        self._connections: Dict[str, ConnectionInfo] = {}
        # Global sequence counter
        self._sequence: int = 0
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_buffer: Dict[str, list] = {}  # user_id -> missed messages
        self._buffer_max_size = 100
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        rooms: Optional[Set[str]] = None
    ) -> str:
        """
        Accept a WebSocket connection and register it.

        Args:
            websocket: The WebSocket connection
            user_id: User identifier for targeting messages
            rooms: Optional set of room names to join

        Returns:
            Connection ID for this connection
        """
        await websocket.accept()

        connection_id = str(uuid4())
        rooms = rooms or set()
        rooms.add(f"user:{user_id}")  # Always join user's personal room

        connection_info = ConnectionInfo(
            websocket=websocket,
            user_id=user_id,
            rooms=rooms
        )

        async with self._lock:
            self._connections[connection_id] = connection_info

            for room in rooms:
                if room not in self._rooms:
                    self._rooms[room] = set()
                self._rooms[room].add(connection_id)

        logger.info(f"WebSocket connected: user={user_id}, conn_id={connection_id}, rooms={rooms}")

        # Send any buffered messages
        await self._send_buffered_messages(connection_id, user_id)

        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """Remove a connection and clean up room memberships."""
        async with self._lock:
            if connection_id not in self._connections:
                return

            connection_info = self._connections[connection_id]

            # Remove from all rooms
            for room in connection_info.rooms:
                if room in self._rooms:
                    self._rooms[room].discard(connection_id)
                    if not self._rooms[room]:
                        del self._rooms[room]

            del self._connections[connection_id]

        logger.info(f"WebSocket disconnected: conn_id={connection_id}")

    async def join_room(self, connection_id: str, room: str) -> None:
        """Add a connection to a room."""
        async with self._lock:
            if connection_id not in self._connections:
                return

            if room not in self._rooms:
                self._rooms[room] = set()
            self._rooms[room].add(connection_id)
            self._connections[connection_id].rooms.add(room)

    async def leave_room(self, connection_id: str, room: str) -> None:
        """Remove a connection from a room."""
        async with self._lock:
            if connection_id not in self._connections:
                return

            if room in self._rooms:
                self._rooms[room].discard(connection_id)
                if not self._rooms[room]:
                    del self._rooms[room]
            self._connections[connection_id].rooms.discard(room)

    async def send_personal(
        self,
        user_id: str,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> int:
        """Send a message to a specific user's connections."""
        return await self.broadcast_to_room(
            f"user:{user_id}",
            message_type,
            payload
        )

    async def broadcast_to_room(
        self,
        room: str,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> int:
        """Broadcast a message to all connections in a room."""
        async with self._lock:
            self._sequence += 1
            sequence = self._sequence

        message = WebSocketMessage(
            type=message_type,
            payload=payload,
            timestamp=datetime.utcnow().isoformat() + "Z",
            sequence=sequence
        )

        sent_count = 0
        failed_connections = []

        async with self._lock:
            connection_ids = self._rooms.get(room, set()).copy()

        for conn_id in connection_ids:
            async with self._lock:
                conn_info = self._connections.get(conn_id)
            if not conn_info:
                continue

            try:
                await conn_info.websocket.send_json(message.model_dump())
                async with self._lock:
                    conn_info.last_sequence = sequence
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send to {conn_id}: {e}")
                failed_connections.append(conn_id)

        # Clean up failed connections
        for conn_id in failed_connections:
            await self.disconnect(conn_id)

        return sent_count

    async def broadcast_all(
        self,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> int:
        """Broadcast a message to all connected clients."""
        async with self._lock:
            self._sequence += 1
            sequence = self._sequence

        message = WebSocketMessage(
            type=message_type,
            payload=payload,
            timestamp=datetime.utcnow().isoformat() + "Z",
            sequence=sequence
        )

        sent_count = 0
        failed_connections = []

        async with self._lock:
            connection_ids = list(self._connections.keys())

        for conn_id in connection_ids:
            async with self._lock:
                conn_info = self._connections.get(conn_id)
            if not conn_info:
                continue

            try:
                await conn_info.websocket.send_json(message.model_dump())
                async with self._lock:
                    conn_info.last_sequence = sequence
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send to {conn_id}: {e}")
                failed_connections.append(conn_id)

        for conn_id in failed_connections:
            await self.disconnect(conn_id)

        return sent_count

    async def _send_buffered_messages(self, connection_id: str, user_id: str) -> None:
        """Send any buffered messages to a newly connected client."""
        if user_id not in self._message_buffer:
            return

        messages = self._message_buffer.pop(user_id, [])

        async with self._lock:
            conn_info = self._connections.get(connection_id)
        if not conn_info:
            return

        for message in messages:
            try:
                await conn_info.websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send buffered message: {e}")
                break

    async def buffer_message(
        self,
        user_id: str,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> None:
        """Buffer a message for offline user delivery."""
        async with self._lock:
            self._sequence += 1
            sequence = self._sequence

        message = WebSocketMessage(
            type=message_type,
            payload=payload,
            timestamp=datetime.utcnow().isoformat() + "Z",
            sequence=sequence
        ).model_dump()

        if user_id not in self._message_buffer:
            self._message_buffer[user_id] = []

        self._message_buffer[user_id].append(message)

        # Limit buffer size
        if len(self._message_buffer[user_id]) > self._buffer_max_size:
            self._message_buffer[user_id] = self._message_buffer[user_id][-self._buffer_max_size:]

    async def start_heartbeat(self) -> None:
        """Start the heartbeat monitoring task."""
        if self._heartbeat_task is not None:
            return
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop_heartbeat(self) -> None:
        """Stop the heartbeat monitoring task."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats and clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                await self._send_heartbeats()
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    async def _send_heartbeats(self) -> None:
        """Send heartbeat to all connections."""
        async with self._lock:
            connection_ids = list(self._connections.keys())

        heartbeat_message = {
            "type": MessageType.HEARTBEAT,
            "payload": {"server_time": datetime.utcnow().isoformat() + "Z"},
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sequence": -1  # Heartbeats don't use sequence
        }

        for conn_id in connection_ids:
            async with self._lock:
                conn_info = self._connections.get(conn_id)
            if not conn_info:
                continue

            try:
                await conn_info.websocket.send_json(heartbeat_message)
            except Exception:
                pass  # Will be cleaned up by stale connection check

    async def _cleanup_stale_connections(self) -> None:
        """Remove connections that haven't responded to heartbeats."""
        stale_threshold = self._heartbeat_interval * 3  # 3 missed heartbeats
        now = datetime.now()
        stale_connections = []

        async with self._lock:
            for conn_id, conn_info in self._connections.items():
                if (now - conn_info.last_heartbeat).total_seconds() > stale_threshold:
                    stale_connections.append(conn_id)

        for conn_id in stale_connections:
            logger.info(f"Cleaning up stale connection: {conn_id}")
            await self.disconnect(conn_id)

    async def handle_heartbeat_ack(self, connection_id: str) -> None:
        """Update last heartbeat time for a connection."""
        async with self._lock:
            if connection_id in self._connections:
                self._connections[connection_id].last_heartbeat = datetime.now()

    def get_connection_count(self) -> int:
        """Get the total number of active connections."""
        return len(self._connections)

    def get_room_count(self, room: str) -> int:
        """Get the number of connections in a specific room."""
        return len(self._rooms.get(room, set()))

    def get_user_connections(self, user_id: str) -> int:
        """Get the number of connections for a specific user."""
        return self.get_room_count(f"user:{user_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        return {
            "total_connections": len(self._connections),
            "total_rooms": len(self._rooms),
            "current_sequence": self._sequence,
            "buffered_users": len(self._message_buffer),
            "rooms": {room: len(conns) for room, conns in self._rooms.items()}
        }


# Global WebSocket manager instance
ws_manager = WebSocketManager()
