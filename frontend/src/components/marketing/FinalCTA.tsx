'use client'

import { Button } from '@/components/ui/button'
import { BookOpen } from 'lucide-react'
import Link from 'next/link'

export default function FinalCTA() {
  return (
    <section className="py-20 bg-gradient-to-br from-indigo-900 via-purple-900 to-blue-900 text-white">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        {/* Icon */}
        <div className="w-16 h-16 bg-white bg-opacity-10 rounded-full flex items-center justify-center mx-auto mb-8">
          <BookOpen className="w-8 h-8 text-white" />
        </div>
        
        {/* Main heading */}
        <h2 className="text-3xl sm:text-4xl font-bold font-serif mb-8 leading-tight">
          A Library Built for How You Think
        </h2>
        
        {/* Main description */}
        <p className="text-xl text-indigo-100 leading-relaxed font-serif mb-12 max-w-3xl mx-auto">
          Alexandria is a space to pause, reflect, and grow. With transparent AI, 
          thoughtful tools, and a supportive community behind every book, you'll find clarity, not noise.
        </p>
        
        {/* Call to Action */}
        <div className="flex flex-col sm:flex-row justify-center items-center gap-4">
          <Link href="/auth/register">
            <Button 
              size="lg"
              className="bg-white text-indigo-900 hover:bg-indigo-50 px-8 py-4 text-lg font-medium rounded-2xl shadow-lg hover:shadow-xl transition-all duration-200"
            >
              Create your free account
            </Button>
          </Link>
          
          <Link href="/catalog-public">
            <Button 
              variant="outline"
              size="lg"
              className="border-2 border-white text-white hover:bg-white hover:text-indigo-900 px-8 py-4 text-lg font-medium rounded-2xl transition-all duration-200"
            >
              Explore the Public Library
            </Button>
          </Link>
        </div>
      </div>
    </section>
  )
}