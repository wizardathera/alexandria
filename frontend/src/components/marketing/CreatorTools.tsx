'use client'

import { Button } from '@/components/ui/button'
import { Upload, Users, DollarSign } from 'lucide-react'

export default function CreatorTools() {
  const features = [
    {
      icon: Upload,
      title: 'Upload your own books or guides and annotate them'
    },
    {
      icon: Users,
      title: 'Create public or private reading spaces'
    },
    {
      icon: DollarSign,
      title: 'Set your own pricing and earn fairly'
    }
  ]

  return (
    <section className="py-20 bg-white">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold font-serif text-gray-900 mb-8">
            Empower Others Through What You Know
          </h2>
          <p className="text-xl text-gray-700 leading-relaxed font-serif max-w-2xl mx-auto">
            Alexandria gives authors, educators, and curators the tools to turn knowledge into community.
          </p>
        </div>

        {/* Features */}
        <div className="space-y-6 mb-12">
          {features.map((feature, index) => (
            <div key={index} className="flex items-start space-x-4">
              <div className="w-10 h-10 bg-amber-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <feature.icon className="w-5 h-5 text-amber-700" />
              </div>
              <div>
                <p className="text-lg font-serif text-gray-700 leading-relaxed">
                  {feature.title}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* CTA */}
        <div className="text-center">
          <div className="flex flex-col sm:flex-row justify-center items-center gap-4">
            <Button 
              className="bg-indigo-700 hover:bg-indigo-800 text-white px-6 py-3 rounded-2xl font-medium"
            >
              Start your creator profile
            </Button>
            <Button 
              variant="ghost"
              className="text-indigo-700 hover:text-indigo-800 font-medium"
            >
              Learn more about educator tools →
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}