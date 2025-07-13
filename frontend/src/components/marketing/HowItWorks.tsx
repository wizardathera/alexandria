'use client'

import { BookOpen, Play, MessageCircle, Users } from 'lucide-react'

export default function HowItWorks() {
  const steps = [
    {
      number: 1,
      icon: BookOpen,
      title: 'Choose a Book'
    },
    {
      number: 2,
      icon: Play,
      title: 'Start Reading'
    },
    {
      number: 3,
      icon: MessageCircle,
      title: 'Ask Hypatia'
    },
    {
      number: 4,
      icon: Users,
      title: 'Join the Conversation'
    }
  ]

  return (
    <section className="py-20 bg-slate-50">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold font-serif text-gray-900">
            Explore Your Curiosities
          </h2>
        </div>

        {/* Steps - Vertical Flow */}
        <div className="space-y-8">
          {steps.map((step, index) => (
            <div key={step.number} className="flex items-start space-x-4">
              {/* Step Icon and Number */}
              <div className="flex items-center justify-center w-12 h-12 bg-amber-100 rounded-full flex-shrink-0">
                <step.icon className="w-6 h-6 text-amber-700" />
              </div>
              
              {/* Step Content */}
              <div className="flex-1">
                <h3 className="text-xl font-semibold font-serif text-gray-900">
                  {step.number}. {step.title}
                </h3>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}