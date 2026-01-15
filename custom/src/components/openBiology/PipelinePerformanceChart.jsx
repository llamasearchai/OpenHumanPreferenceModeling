'use client'
import React from 'react'
import dynamic from 'next/dynamic'
import CardHeader from '@/components/shared/CardHeader'
import CardLoader from '@/components/shared/CardLoader'
import useCardTitleActions from '@/hooks/useCardTitleActions'
import { pipelinePerformance } from '@/utils/openBiologyData'

const ReactApexChart = dynamic(() => import('react-apexcharts'), { ssr: false });

const colors = ["#3454d1", "#25b865", "#ffa21d", "#d13b4c", "#5e72e4"]

const PipelinePerformanceChart = ({ data = pipelinePerformance }) => {
  const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();

  if (isRemoved) {
    return null;
  }

  const chartOptions = {
    chart: {
      stacked: false,
      toolbar: { show: false },
    },
    stroke: {
      width: [2, 2, 3, 3, 2],
      curve: "smooth",
      lineCap: "round",
    },
    plotOptions: {
      bar: {
        borderRadius: 6,
        columnWidth: "26%",
      }
    },
    colors,
    series: data.series,
    fill: {
      opacity: [0.85, 0.85, 0.35, 0.35, 0.15]
    },
    markers: { size: 0 },
    xaxis: {
      categories: data.categories,
      labels: { style: { fontSize: "11px", colors: "#A0ACBB" } },
      axisBorder: { show: false },
      axisTicks: { show: false },
    },
    yaxis: {
      labels: {
        formatter: (value) => `${value}`,
        style: { colors: "#A0ACBB" }
      },
      title: {
        text: "Runs completed",
        style: { fontSize: "12px" }
      }
    },
    grid: {
      strokeDashArray: 4,
      padding: { left: 12, right: 12 }
    },
    legend: {
      show: true,
      position: "top",
      fontSize: "12px"
    },
    tooltip: {
      y: {
        formatter: (value) => `${value} runs`
      }
    }
  }

  return (
    <div className="col-xxl-4">
      <div className={`card stretch stretch-full ${isExpanded ? "card-expand" : ""} ${refreshKey ? "card-loading" : ""}`}>
        <CardHeader title={"Pipeline throughput"} refresh={handleRefresh} remove={handleDelete} expanded={handleExpand} />
        <div className="card-body custom-card-action">
          <ReactApexChart
            options={chartOptions}
            series={chartOptions.series}
            type='line'
            height={350}
          />
          <p className="fs-12 text-muted mb-0">Runs per month across integrated OpenBiology modules.</p>
        </div>
        <CardLoader refreshKey={refreshKey} />
      </div>
    </div>
  )
}

export default PipelinePerformanceChart
