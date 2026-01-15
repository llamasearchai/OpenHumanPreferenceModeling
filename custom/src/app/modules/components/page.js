'use client'
import React from 'react'
import PageHeader from '@/components/shared/pageHeader/PageHeader'
import PageHeaderDate from '@/components/shared/pageHeader/PageHeaderDate'
import SiteOverviewStatistics from '@/components/widgetsStatistics/SiteOverviewStatistics'
import ModuleGrid from '@/components/openBiology/ModuleGrid'
import PipelinePerformanceChart from '@/components/openBiology/PipelinePerformanceChart'
import ActivityTable from '@/components/openBiology/ActivityTable'
import Schedule from '@/components/widgetsList/Schedule'
import ModuleResourceList from '@/components/openBiology/ModuleResourceList'
import Project from '@/components/widgetsList/Project'
import Progress from '@/components/widgetsList/Progress'
import { openBiologyModules, overviewMetrics, upcomingActions } from '@/utils/openBiologyData'
import { projectsData } from '@/utils/fackData/projectsData'

const ComponentsGallery = () => {
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
        <div className="alert alert-secondary border-0 shadow-sm" role="alert">
          Gallery of core OpenBiology widgets wired to module data. Use this as a reference to compose new pages fast.
        </div>
        <div className="row g-4">
          <SiteOverviewStatistics data={overviewMetrics} />
          <ModuleGrid modules={openBiologyModules} title="Modules cards" />
          <PipelinePerformanceChart />
          <ModuleResourceList />
          <ActivityTable title="Activity table" />
          <Schedule title="Handoffs" data={scheduleData} showFooter={false} />
          <Project title="Projects / Runs" data={projectsData.runningProjects} cardYSpaceClass="hrozintioal-card" borderShow={true} footerText="Upcoming runs" />
          <Progress title="Teams on-call" footerShow={false} />
        </div>
      </div>
    </>
  )
}

export default ComponentsGallery
