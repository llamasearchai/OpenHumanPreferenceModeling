'use client'
import React from 'react'
import Link from 'next/link'
import CardHeader from '@/components/shared/CardHeader'
import CardLoader from '@/components/shared/CardLoader'
import useCardTitleActions from '@/hooks/useCardTitleActions'
import { runActivity } from '@/utils/openBiologyData'

const statusColor = {
  Running: "primary",
  Ingesting: "info",
  Review: "warning",
  Attention: "danger",
  Done: "success"
}

const ActivityTable = ({ data = runActivity, title = "Latest activity" }) => {
  const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();

  if (isRemoved) {
    return null;
  }

  return (
    <div className="col-xxl-12">
      <div className={`card stretch stretch-full ${isExpanded ? "card-expand" : ""} ${refreshKey ? "card-loading" : ""}`}>
        <CardHeader title={title} refresh={handleRefresh} remove={handleDelete} expanded={handleExpand} />
        <div className="card-body custom-card-action p-0">
          <div className="table-responsive">
            <table className="table table-hover mb-0">
              <thead>
                <tr className="border-b">
                  <th scope="row">Module</th>
                  <th>Work item</th>
                  <th>Stage</th>
                  <th>Status</th>
                  <th>Owner</th>
                  <th className="text-end">ETA/Notes</th>
                </tr>
              </thead>
              <tbody>
                {data.map(({ id, module, item, stage, status, owner, eta, moduleKey }) => (
                  <tr key={id} className="chat-single-item">
                    <td className="fw-semibold text-dark">
                      {moduleKey ? <Link href={`/modules/${moduleKey}`}>{module}</Link> : module}
                    </td>
                    <td>
                      <div className="d-block">
                        <span className="d-block text-dark">{item}</span>
                        <span className="fs-12 text-muted">{stage}</span>
                      </div>
                    </td>
                    <td className="text-muted">{stage}</td>
                    <td>
                      <span className={`badge bg-soft-${statusColor[status] || "secondary"} text-${statusColor[status] || "secondary"}`}>{status}</span>
                    </td>
                    <td>{owner}</td>
                    <td className="text-end">
                      {moduleKey ? (
                        <Link href={`/modules/${moduleKey}`} className="text-muted fs-12">ETA: {eta}</Link>
                      ) : (
                        <span className="text-muted fs-12">{eta}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <CardLoader refreshKey={refreshKey} />
      </div>
    </div>
  )
}

export default ActivityTable
