'use client'
import React from 'react'
import Link from 'next/link'
import CardHeader from '@/components/shared/CardHeader'
import CardLoader from '@/components/shared/CardLoader'
import useCardTitleActions from '@/hooks/useCardTitleActions'
import { moduleKpis, moduleResources, openBiologyModules } from '@/utils/openBiologyData'

const ModuleDetail = ({ moduleKey }) => {
  const module = openBiologyModules.find((m) => m.key === moduleKey)
  const kpi = moduleKpis[moduleKey]
  const resource = moduleResources[moduleKey]
  const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();

  if (!module || !kpi || isRemoved) {
    return null
  }

  return (
    <div className={`card stretch stretch-full ${isExpanded ? "card-expand" : ""} ${refreshKey ? "card-loading" : ""}`}>
      <CardHeader title={module.name} refresh={handleRefresh} remove={handleDelete} expanded={handleExpand} />
      <div className="card-body custom-card-action">
        <div className="d-flex align-items-center justify-content-between flex-wrap gap-3 mb-3">
          <div>
            <div className="fs-12 text-muted text-uppercase">{module.status}</div>
            <h4 className="text-dark mb-1">{module.name}</h4>
            <p className="text-muted fs-12 mb-0">{module.focus}</p>
          </div>
          <div className="text-end">
            <div className="fs-12 text-muted">Owner</div>
            <div className="fw-semibold text-dark">{module.owner}</div>
            <div className="fs-12 text-muted">Health: {module.health}</div>
          </div>
        </div>
        <div className="mb-4">
          <div className="d-flex align-items-center mb-2">
            <div className="progress w-100 me-3 ht-8">
              <div className="progress-bar bg-primary" role="progressbar" style={{ width: `${module.progress}%` }} aria-valuenow={module.progress} aria-valuemin="0" aria-valuemax="100"></div>
            </div>
            <span className="fw-semibold text-dark">{module.progress}%</span>
          </div>
          <div className="fs-12 text-muted d-flex gap-3 flex-wrap">
            <span>Active runs: <strong className="text-dark">{module.activeRuns}</strong></span>
            <span>Last run: <strong className="text-dark">{module.lastRun}</strong></span>
            <span>Dependencies: {module.dependencies.length ? module.dependencies.join(", ") : "none"}</span>
          </div>
        </div>
        <div className="row g-3 mb-4">
          {kpi.stats.map(({ label, value }) => (
            <div key={label} className="col-md-3 col-sm-6">
              <div className="p-3 border border-dashed rounded-3 h-100">
                <div className="fs-12 text-muted">{label}</div>
                <div className="fw-semibold text-dark fs-5">{value}</div>
              </div>
            </div>
          ))}
        </div>
        <div className="row g-4">
          <div className="col-lg-6">
            <h6 className="fw-semibold text-dark">What to ship next</h6>
            <ul className="list-unstyled m-0">
              {kpi.backlog.map((item, idx) => (
                <li key={idx} className="d-flex gap-2 align-items-start py-2">
                  <span className="text-primary mt-1">•</span>
                  <span className="text-muted fs-12">{item}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="col-lg-6">
            <h6 className="fw-semibold text-dark">Checkpoints</h6>
            <ul className="list-unstyled m-0">
              {kpi.checkpoints.map((item, idx) => (
                <li key={idx} className="d-flex gap-2 align-items-start py-2">
                  <span className="text-success mt-1">•</span>
                  <span className="text-muted fs-12">{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        {resource && (
          <div className="mt-4">
            <h6 className="fw-semibold text-dark">Resources</h6>
            <div className="fs-12 text-muted">Repo path: <span className="text-dark">{resource.repo}</span></div>
            <div className="fs-12 text-muted">Docs: <span className="text-dark">{resource.docs}</span></div>
            <div className="d-flex flex-wrap gap-2 mt-2">
              {resource.pages.map((p) => (
                <span key={p} className="badge bg-soft-primary text-primary">{p}</span>
              ))}
            </div>
          </div>
        )}
        <div className="mt-4 d-flex justify-content-between align-items-center">
          <Link className="btn btn-primary" href="#">Trigger run</Link>
          <Link className="fs-12 text-uppercase fw-bold" href="#">Open SOPs</Link>
        </div>
      </div>
      <CardLoader refreshKey={refreshKey} />
    </div>
  )
}

export default ModuleDetail
