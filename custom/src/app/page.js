import React from 'react'
import PageHeader from '@/components/shared/pageHeader/PageHeader'
import PageHeaderDate from '@/components/shared/pageHeader/PageHeaderDate'
import SiteOverviewStatistics from '@/components/widgetsStatistics/SiteOverviewStatistics'
import Schedule from '@/components/widgetsList/Schedule'
import DuplicateLayout from './duplicateLayout'
import ModuleGrid from '@/components/openBiology/ModuleGrid'
import PipelinePerformanceChart from '@/components/openBiology/PipelinePerformanceChart'
import ActivityTable from '@/components/openBiology/ActivityTable'
import ModuleResourceList from '@/components/openBiology/ModuleResourceList'
import { openBiologyModules, overviewMetrics, upcomingActions } from '@/utils/openBiologyData'

const Home = () => {
  const scheduleData = upcomingActions.map(({ id, title, participants, date, color }) => ({
    id,
    schedule_name: title,
    team_members: participants,
    date,
    color
  }))

  return (
    <DuplicateLayout>
      <PageHeader >
        <PageHeaderDate />
      </PageHeader>
      <div className='main-content'>
        <div className="alert alert-primary border-0 shadow-sm" role="alert">
          Unified OpenBiology brings Biomarker Analysis, Matrices, Bioprocess Automation, Genomic Sequencing, and Imaging Agents into one control plane tailored to lab operations.
        </div>
        <div className='row'>
          <SiteOverviewStatistics data={overviewMetrics} />
          <ModuleGrid modules={openBiologyModules} />
          <PipelinePerformanceChart />
          <ModuleResourceList />
          <Schedule title={"Upcoming handoffs"} data={scheduleData} showFooter={false} />
          <ActivityTable title="Cross-module activity" />
        </div>
      </div>
    </DuplicateLayout>
  )
}

export default Home
