'use client'
import React from 'react'
import PageHeader from '@/components/shared/pageHeader/PageHeader'
import PageHeaderDate from '@/components/shared/pageHeader/PageHeaderDate'
import ModuleGrid from '@/components/openBiology/ModuleGrid'
import ActivityTable from '@/components/openBiology/ActivityTable'
import PipelinePerformanceChart from '@/components/openBiology/PipelinePerformanceChart'
import Schedule from '@/components/widgetsList/Schedule'
import ModuleResourceList from '@/components/openBiology/ModuleResourceList'
import { openBiologyModules, upcomingActions, runActivity } from '@/utils/openBiologyData'

const ModulesHome = () => {
  const scheduleData = upcomingActions.map(({ id, title, participants, date, color }) => ({
    id,
    schedule_name: title,
    team_members: participants,
    date,
    color
  }))

  return (
    <>
      <PageHeader>
        <PageHeaderDate />
      </PageHeader>
      <div className='main-content'>
        <div className="alert alert-info border-0 shadow-sm" role="alert">
          Module directory for all OpenBiology surfaces. Jump into a module, check throughput, or scan upcoming handoffs.
        </div>
        <div className='row g-4'>
          <ModuleGrid modules={openBiologyModules} title="Modules overview" />
          <PipelinePerformanceChart />
          <ModuleResourceList />
          <ActivityTable data={runActivity} title="Live activity across modules" />
          <Schedule title="Planned handoffs" data={scheduleData} showFooter={false} />
        </div>
      </div>
    </>
  )
}

export default ModulesHome
