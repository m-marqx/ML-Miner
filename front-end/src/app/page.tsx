"use client"

import { useState, useEffect } from "react"
import {
  ArrowRight,
  TrendingUp,
  Zap,
  Sparkles,
  CheckCircle,
  Circle,
  Clock,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import React from "react"
import { redirect } from "next/navigation"

const roadmapData = [
  {
    id: 1,
    title: "Platform Launch",
    description: "Core trading infrastructure and basic portfolio management",
    status: "completed",
    date: "Q2 2025",
    features: ["Real-time trading", "Basic portfolio tracking", "AI predictions"],
  },
  {
    id: 2,
    title: "Add New Networks",
    description: "additional blockchain networks and cross-chain support",
    status: "upcoming",
    date: "Q3 2025",
    features: ["Add Multi-chain support"],
  },
]

export default function HomePage() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const [isLoaded, setIsLoaded] = useState(false)
  const [activeFeature, setActiveFeature] = useState(0)
  const [activeRoadmapItem, setActiveRoadmapItem] = useState(2) // Current item

  useEffect(() => {
    setIsLoaded(true)

    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth) * 100,
        y: (e.clientY / window.innerHeight) * 100,
      })
    }

    window.addEventListener("mousemove", handleMouseMove)
    return () => window.removeEventListener("mousemove", handleMouseMove)
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % 3)
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="w-5 h-5" style={{ color: "#10b981" }} />
      case "current":
        return <Clock className="w-5 h-5" style={{ color: "#d87a16" }} />
      default:
        return <Circle className="w-5 h-5" style={{ color: "rgba(229, 231, 235, 0.45)" }} />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "#10b981"
      case "current":
        return "#d87a16"
      default:
        return "rgba(229, 231, 235, 0.25)"
    }
  }

  return (
    <div className="min-h-screen overflow-hidden relative bg-background-color">
      {/* Navigation */}
      <nav className="absolute right-0 top-0 z-50 flex items-center justify-between p-6 lg:px-8">
        <div className="flex items-center space-x-4 text-black/65 font-medium">
          <Button
            className="transition-all duration-200 hover:shadow-lg bg-main-color text-primary-size"
            onClick={() => redirect("/swap")}
          >
            Launch dApp
          </Button>
        </div>
      </nav>

      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Gradient Orbs */}
        <div
          className="absolute w-96 h-96 rounded-full blur-3xl transition-transform duration-1000 ease-out opacity-15"
          style={{
            background: "linear-gradient(135deg, #d87a16, #f59e0b)",
            transform: `translate(${mousePosition.x * 0.02}px, ${mousePosition.y * 0.02}px)`,
            left: "10%",
            top: "20%",
          }}
        />
        <div
          className="absolute w-96 h-96 rounded-full blur-3xl transition-transform duration-1000 ease-out opacity-10"
          style={{
            background: "linear-gradient(135deg, #7c3aed, #3b82f6)",
            transform: `translate(${mousePosition.x * -0.02}px, ${mousePosition.y * -0.02}px)`,
            right: "10%",
            bottom: "20%",
          }}
        />

        {/* Geometric Grid */}
        <div
          className="absolute inset-0 opacity-5"
          style={{
            backgroundImage: `linear-gradient(rgba(216, 122, 22, 0.2) 1px, transparent 1px),
                           linear-gradient(90deg, rgba(216, 122, 22, 0.2) 1px, transparent 1px)`,
            backgroundSize: "60px 60px",
          }}
        />
      </div>

      {/* Hero Content */}
      <main className="relative z-10 flex flex-col items-center justify-center min-h-[calc(100vh-80px)] px-6">
        <div className="max-w-7xl mx-auto text-center">

          {/* Main Heading */}
          <h1
            className={`text-5xl md:text-7xl lg:text-8xl font-bold mb-2 transition-all duration-1000 delay-200 ${
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
            }`}
          >
            <span
              className="block bg-gradient-to-r from-[#d87a16] via-[#f59e0b] to-[#fbbf24] bg-clip-text text-transparent text-9xl leading-relaxed"
            >
              MX Swap
            </span>
          </h1>

          {/* Description */}
          <p
            className={`text-xl md:text-2xl max-w-4xl mx-auto mb-12 transition-all duration-1000 delay-400 ${
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
            }`}
            style={{
              color: "rgba(229, 231, 235, 0.65)",
              lineHeight: "calc(1.5rem + 0.5rem)",
            }}
          >
            The next-generation decentralized exchange platform, combining KyberSwap and VeloraSwap best prices to offer the best trading experience. 
            With a portfolio visualizer and AI Signals to help you make informed decisions.
          </p>

          {/* Feature Cards */}
          <div
            className={`grid md:grid-cols-3 gap-6 max-w-5xl mx-auto transition-all duration-1000 delay-1200 my-10 ${
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
            }`}
          >
            {[
              {
                icon: Zap,
                title: "Advanced Agreggator",
                description: "Get the best prices across multiple DEXs with our smart routing technology",
                color: "#d87a16",
                bgColor: "rgba(216, 122, 22, 0.15)",
                borderColor: "rgba(216, 122, 22, 0.3)",
                active: activeFeature === 0,
              },
              {
                icon: Sparkles,
                title: "AI Signals",
                description: "Leverage AI-driven insights to optimize your trading strategies and maximize returns",
                color: "#7c3aed",
                bgColor: "rgba(124, 58, 237, 0.15)",
                borderColor: "rgba(124, 58, 237, 0.3)",
                active: activeFeature === 1,
              },
              {
                icon: TrendingUp,
                title: "Portfolio Visualizer",
                description: "Visualize your portfolio performance in real-time",
                color: "#3b82f6",
                bgColor: "rgba(59, 130, 246, 0.15)",
                borderColor: "rgba(59, 130, 246, 0.3)",
                active: activeFeature === 2,
              },
            ].map((feature, index) => (
              <Card
                key={index}
                className={`group border backdrop-blur-sm transition-all duration-500 hover:scale-105 cursor-pointer ${
                  feature.active ? "scale-105 shadow-lg bg-zinc-900" : "bg-gray-950 shadow-sm"
                }`}
                style={{
                  // backgroundColor: feature.active ? "rgba(31, 41, 55, 0.8)" : "rgba(31, 41, 55, 0.4)",
                  borderColor: feature.active ? feature.borderColor : "rgba(75, 85, 99, 0.3)",
                  boxShadow: feature.active
                    ? `0 10px 25px ${feature.color}20, 0 0 20px ${feature.color}10`
                    : "0 4px 6px rgba(0, 0, 0, 0.1)",
                }}
                onMouseEnter={() => setActiveFeature(index)}
              >
                <CardContent className="p-8">
                  <div
                    className="w-14 h-14 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300"
                    style={{ backgroundColor: feature.bgColor }}
                  >
                    <feature.icon className="w-7 h-7" style={{ color: feature.color }} />
                  </div>
                  <h3
                    className="text-xl font-semibold mb-4"
                    style={{
                      color: "rgba(229, 231, 235, 0.85)",
                      lineHeight: "calc(1.25rem + 0.5rem)",
                    }}
                  >
                    {feature.title}
                  </h3>
                  <p
                    className="text-sm leading-relaxed"
                    style={{
                      color: "rgba(229, 231, 235, 0.65)",
                      lineHeight: "calc(0.875rem + 0.5rem)",
                    }}
                  >
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Dynamic Roadmap Component */}
          <div
            className={`mb-16 transition-all duration-1000 delay-600 ${
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
            }`}
          >
            <div className="mb-8">
              <h2
                className="text-2xl md:text-3xl font-bold mb-4"
                style={{
                  color: "rgba(229, 231, 235, 0.85)",
                  lineHeight: "calc(2rem + 0.5rem)",
                }}
              >
                Development Roadmap
              </h2>
              <p
                className="text-lg max-w-2xl mx-auto"
                style={{
                  color: "rgba(229, 231, 235, 0.65)",
                  lineHeight: "calc(1.125rem + 0.5rem)",
                }}
              >
                Track our journey as we build the most advanced financial platform
              </p>
            </div>

            {/* Roadmap Timeline */}
            <div className="relative max-w-6xl mx-auto">
              {/* Timeline Line */}
              <div className="absolute left-1/2 transform -translate-x-1/2 w-1 h-full hidden md:block">
                <div
                  className="w-full h-full rounded-full"
                  style={{
                    background: `linear-gradient(to bottom,
                      #10b981 0%,
                      #10b981 50%,
                      rgba(229, 231, 235, 0.25) 50%,
                      rgba(229, 231, 235, 0.25) 100%)`,
                  }}
                />
              </div>

              {/* Roadmap Items */}
              <div className="space-y-8 md:space-y-12">
                {roadmapData.map((item, index) => (
                  <div
                    key={item.id}
                    className={`relative flex flex-col md:flex-row items-center transition-all duration-500 ${
                      index % 2 === 0 ? "md:flex-row" : "md:flex-row-reverse"
                    }`}
                    onMouseEnter={() => setActiveRoadmapItem(index)}
                  >
                    {/* Timeline Node */}
                    <div className="absolute left-1/2 transform -translate-x-1/2 z-10 hidden md:block">
                      <div
                        className="w-12 h-12 rounded-full border-4 flex items-center justify-center transition-all duration-300 hover:scale-110"
                        style={{
                          backgroundColor: "#0a0a0b",
                          borderColor: getStatusColor(item.status),
                          boxShadow: `0 0 20px ${getStatusColor(item.status)}40`,
                        }}
                      >
                        {getStatusIcon(item.status)}
                      </div>
                    </div>

                    {/* Content Card */}
                    <div
                      className={`w-full md:w-5/12 ${index % 2 === 0 ? "md:mr-auto md:pr-8" : "md:ml-auto md:pl-8"}`}
                    >
                      <Card
                        className={`backdrop-blur-sm border transition-all duration-300 hover:scale-105 cursor-pointer ${
                          activeRoadmapItem === index ? `scale-105 shadow-lg bg-zinc-900 border-[${getStatusColor(item.status)}60]` : "bg-gray-950 shadow-sm"
                        }`}
                        style={{
                          borderColor:
                            activeRoadmapItem === index ? `${getStatusColor(item.status)}60` : "",
                          boxShadow:
                            activeRoadmapItem === index
                              ? `0 10px 25px ${getStatusColor(item.status)}20, 0 0 20px ${getStatusColor(
                                  item.status,
                                )}10`
                              : "0 4px 6px rgba(0, 0, 0, 0.1)",
                        }}
                      >
                        <CardContent className="p-6">
                          {/* Mobile Timeline Node */}
                          <div className="flex items-center mb-4 md:hidden">
                            {getStatusIcon(item.status)}
                            <span
                              className="ml-3 text-sm font-medium"
                              style={{
                                color: getStatusColor(item.status),
                                lineHeight: "calc(0.875rem + 0.5rem)",
                              }}
                            >
                              {item.date}
                            </span>
                          </div>

                          {/* Date Badge */}
                          <div className="hidden md:block mb-4">
                            <span
                              className="inline-block px-3 py-1 rounded-full text-xs font-medium"
                              style={{
                                backgroundColor: `${getStatusColor(item.status)}20`,
                                color: getStatusColor(item.status),
                                lineHeight: "calc(0.75rem + 0.5rem)",
                              }}
                            >
                              {item.date}
                            </span>
                          </div>

                          {/* Title */}
                          <h3
                            className="text-xl font-bold mb-3"
                            style={{
                              color: "rgba(229, 231, 235, 0.85)",
                              lineHeight: "calc(1.25rem + 0.5rem)",
                            }}
                          >
                            {item.title}
                          </h3>

                          {/* Description */}
                          <p
                            className="text-sm mb-4"
                            style={{
                              color: "rgba(229, 231, 235, 0.65)",
                              lineHeight: "calc(0.875rem + 0.5rem)",
                            }}
                          >
                            {item.description}
                          </p>

                          {/* Features */}
                          <div className="space-y-2">
                            {item.features.map((feature, featureIndex) => (
                              <div key={featureIndex} className="flex items-center space-x-2">
                                <div
                                  className="w-1.5 h-1.5 rounded-full"
                                  style={{ backgroundColor: getStatusColor(item.status) }}
                                />
                                <span
                                  className="text-xs"
                                  style={{
                                    color: "rgba(229, 231, 235, 0.45)",
                                    lineHeight: "calc(0.75rem + 0.5rem)",
                                  }}
                                >
                                  {feature}
                                </span>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* CTA Buttons */}
          <div
            className={`flex flex-col sm:flex-row gap-4 justify-center mb-16 transition-all duration-1000 delay-1000 ${
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
            }`}
          >
            <Button
              className="px-8 py-4 text-lg font-medium group transition-all duration-200 hover:shadow-xl"
              style={{
                backgroundColor: "#d87a16",
                color: "#0a0a0b",
                lineHeight: "calc(1.125rem + 0.5rem)",
                boxShadow: "0 10px 25px rgba(216, 122, 22, 0.3)",
              }}
              onClick={() => redirect("/swap")}
            >
              Launch dApp
              <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform duration-200" />
            </Button>
          </div>

        </div>

      </main>
    </div>
  )
}
