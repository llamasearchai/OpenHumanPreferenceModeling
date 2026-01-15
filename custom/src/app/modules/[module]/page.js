'use client'
import React, { useMemo } from 'react'
import PageHeader from '@/components/shared/pageHeader/PageHeader'
import PageHeaderDate from '@/components/shared/pageHeader/PageHeaderDate'
import ModuleDetail from '@/components/openBiology/ModuleDetail'
import ActivityTable from '@/components/openBiology/ActivityTable'
import Schedule from '@/components/widgetsList/Schedule'
import { openBiologyModules, runActivity, upcomingActions } from '@/utils/openBiologyData'

const ModulePage = ({ params }) => {
  const moduleKey = params.module
  const moduleMeta = openBiologyModules.find((m) => m.key === moduleKey)

  const activity = useMemo(() => {
    if (!moduleMeta) return []
    return runActivity.filter((item) => item.moduleKey === moduleMeta.key)
  }, [moduleMeta])

  const scheduleData = useMemo(() => {
    if (!moduleMeta) return []
    return upcomingActions
      .filter((item) => item.moduleKey === moduleMeta.key || item.title.toLowerCase().includes(moduleMeta.name.split(" ")[0].toLowerCase()) || item.owner.toLowerCase().includes(moduleMeta.owner.toLowerCase()))
      .map(({ id, title, participants, date, color }) => ({
        id,
        schedule_name: title,
        team_members: participants,
        date,
        color
      }))
  }, [moduleMeta])

  if (!moduleMeta) {
    return (
      <>
        <PageHeader>
          <PageHeaderDate />
        </PageHeader>
        <div className='main-content'>
          <div className="alert alert-warning">Module not found.</div>
        </div>
      </>
    )
  }

  return (
    <>
      <PageHeader>
        <PageHeaderDate />
      </PageHeader>
      <div className='main-content'>
        <div className='row g-4'>
          <div className="col-12">
            <ModuleDetail moduleKey={moduleKey} />
          </div>
          <div className="col-xxl-8">
            <ActivityTable data={activity.length ? activity : runActivity} title="Module activity" />
          </div>
          <div className="col-xxl-4">
            <Schedule title="Next handoffs" data={scheduleData.length ? scheduleData : upcomingActions.map(({ id, title, participants, date, color }) => ({ id, schedule_name: title, team_members: participants, date, color }))} showFooter={false} />
          </div>
        </div>
      </div>
    </>
  )
}

export default ModulePage
