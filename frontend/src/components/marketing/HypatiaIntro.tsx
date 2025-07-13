'use client'

import { Button } from '@/components/ui/button'
import { Sparkles, Brain, User, Lightbulb } from 'lucide-react'

export default function HypatiaIntro() {
  const features = [
    {
      icon: Brain,
      title: 'Context-Aware',
      description: 'Understands the full context of what you\'re reading, not just isolated questions.'
    },
    {
      icon: User,
      title: 'Personalized to You',
      description: 'Learns your interests and adapts to your unique learning journey.'
    },
    {
      icon: Lightbulb,
      title: 'Insight-Oriented',
      description: 'Connects ideas across your library to spark new understanding.'
    }
  ]

  return (
    <section className="py-20 bg-white">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          {/* Left Column - Content */}
          <div>
            <h2 className="text-3xl sm:text-4xl font-bold font-serif text-gray-900 mb-8">
              Read With Hypatia
            </h2>
            
            <div className="text-lg text-gray-700 leading-relaxed font-serif mb-8">
              <p>
                Hypatia is your personal learning companion inside Alexandria. She helps you slow down, 
                reflect, and engage more deeply with every book you read. Ask her to explain difficult 
                passages, connect themes, or offer journaling prompts—always with transparency and context. 
                Hypatia supports your thinking without replacing it.
              </p>
            </div>
          </div>
          
          {/* Right Column - Features */}
          <div className="space-y-6">
            {features.map((feature, index) => (
              <div key={index} className="flex items-start space-x-4">
                <div className="w-10 h-10 bg-amber-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                  <feature.icon className="w-5 h-5 text-amber-700" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold font-serif text-gray-900 mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 font-serif leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}