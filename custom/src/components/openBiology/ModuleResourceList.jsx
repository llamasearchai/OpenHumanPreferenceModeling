'use client'
import React from 'react'
import CardHeader from '@/components/shared/CardHeader'
import CardLoader from '@/components/shared/CardLoader'
import useCardTitleActions from '@/hooks/useCardTitleActions'
import { moduleResources, openBiologyModules } from '@/utils/openBiologyData'

const ModuleResourceList = () => {
  const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();

  if (isRemoved) return null

  return (
    <div className="col-xxl-4">
      <div className={`card stretch stretch-full ${isExpanded ? "card-expand" : ""} ${refreshKey ? "card-loading" : ""}`}>
        <CardHeader title={"Module resources"} refresh={handleRefresh} remove={handleDelete} expanded={handleExpand} />
        <div className="card-body">
          <div className="vstack gap-3">
            {openBiologyModules.map((mod) => {
              const resource = moduleResources[mod.key]
              return (
                <div key={mod.key} className="p-3 border border-dashed rounded-3">
                  <div className="d-flex align-items-center justify-content-between">
                    <div>
                      <div className="fw-semibold text-dark">{mod.name}</div>
                      <div className="fs-12 text-muted">{mod.focus}</div>
                    </div>
                    <span className="badge bg-gray-200 text-dark">{mod.owner}</span>
                  </div>
                  {resource && (
                    <>
                      <div className="fs-12 text-muted mt-2">Repo: <span className="text-dark">{resource.repo}</span></div>
                      <div className="fs-12 text-muted">Docs: <span className="text-dark">{resource.docs}</span></div>
                      <div className="d-flex flex-wrap gap-2 mt-2">
                        {resource.pages.map((p) => (
                          <span key={p} className="badge bg-soft-primary text-primary">{p}</span>
                        ))}
                      </div>
                    </>
                  )}
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

export default ModuleResourceList
