'use client'
import React from 'react'
import DuplicateLayout from '../duplicateLayout'

const ModulesLayout = ({ children }) => {
  return (
    <DuplicateLayout>
      {children}
    </DuplicateLayout>
  )
}

export default ModulesLayout
