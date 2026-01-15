'use client'
import React from 'react'
import Link from 'next/link'
import CardHeader from '@/components/shared/CardHeader'
import CardLoader from '@/components/shared/CardLoader'
import useCardTitleActions from '@/hooks/useCardTitleActions'
import { openBiologyModules } from '@/utils/openBiologyData'

const statusColorMap = {
  Operational: "success",
  Running: "primary",
  Ingesting: "info",
  "QC gating": "warning",
  Attention: "danger",
  Stable: "success"
}

const ModuleGrid = ({ modules = openBiologyModules, title = "OpenBiology Modules" }) => {
  const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();

  if (isRemoved) {
    return null;
  }

  return (
    <div className="col-xxl-8">
      <div className={`card stretch stretch-full ${isExpanded ? "card-expand" : ""} ${refreshKey ? "card-loading" : ""}`}>
        <CardHeader title={title} refresh={handleRefresh} remove={handleDelete} expanded={handleExpand} />
        <div className="card-body custom-card-action">
          <div className="row g-3">
            {modules.map(({ key, name, status, progress, focus, dependencies, health, activeRuns, lastRun, owner }) => {
              const badgeColor = statusColorMap[status] || "primary";
              return (
                <div key={key} className="col-lg-6">
                  <div className="p-3 border border-dashed rounded-3 h-100">
                    <div className="d-flex align-items-center justify-content-between mb-2">
                      <div>
                        <div className="fs-6 fw-bold text-dark">{name}</div>
                        <div className="fs-12 text-muted">Owner: {owner}</div>
                      </div>
                      <div className="text-end">
                        <span className={`badge bg-soft-${badgeColor} text-${badgeColor}`}>{status}</span>
                        <div className="fs-12 text-muted mt-1">Health: {health}</div>
                      </div>
                    </div>
                    <p className="text-muted fs-12 mb-3">{focus}</p>
                    <div className="d-flex align-items-center mb-3">
                      <div className="progress w-100 me-3 ht-6">
                        <div className={`progress-bar bg-${badgeColor}`} role="progressbar" style={{ width: `${progress}%` }} aria-valuenow={progress} aria-valuemin="0" aria-valuemax="100"></div>
                      </div>
                      <span className="fs-12 text-muted">{progress}%</span>
                    </div>
                    <div className="d-flex align-items-center justify-content-between fs-12 text-muted">
                      <div>Active runs: <span className="text-dark fw-semibold">{activeRuns}</span></div>
                      <div>Last run: <span className="text-dark fw-semibold">{lastRun}</span></div>
                    </div>
                    <div className="d-flex align-items-center flex-wrap gap-2 mt-3">
                      {dependencies.length ? (
                        dependencies.map((dep) => (
                          <span key={dep} className="badge bg-gray-200 text-dark">{dep}</span>
                        ))
                      ) : (
                        <span className="text-muted fs-12">No upstream dependencies</span>
                      )}
                    </div>
                    <div className="mt-3 d-flex justify-content-between">
                      <Link className="fs-12 text-uppercase fw-bold" href={`/modules/${key}`}>Open detail</Link>
                      <Link className="fs-12 text-muted" href="#">View logs</Link>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
        <CardLoader refreshKey={refreshKey} />
      </div>
    </div>
  )
}

export default ModuleGrid
