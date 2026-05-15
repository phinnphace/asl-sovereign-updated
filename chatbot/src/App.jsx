import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Home from './Home.jsx'
import Library from './Library.jsx'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/library" element={<Library />} />
    </Routes>
  )
}
