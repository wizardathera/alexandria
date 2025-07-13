'use client'

import { Button } from '@/components/ui/button'
import { BookOpen, MessageCircle, Users } from 'lucide-react'
import Link from 'next/link'

export default function Hero() {
  return (
    <section className="relative bg-gradient-to-br from-slate-50 via-white to-blue-50 pt-20 pb-24 overflow-hidden">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          {/* Main heading - Alexandria format */}
          <h1 className="font-serif text-gray-900 leading-tight mb-8">
            <div className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-normal mb-2">
              Welcome to the Library of
            </div>
            <div className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-indigo-700">
              ALEXANDRIA
            </div>
          </h1>
          
          {/* Subtitle */}
          <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-12 leading-relaxed font-serif">
            Your personal library for nurturing curiosity and personal growth.
          </p>
          
          {/* CTA Button */}
          <div className="mb-16">
            <Link href="/auth/register">
              <Button 
                size="lg" 
                className="bg-indigo-700 hover:bg-indigo-800 text-white px-8 py-4 text-lg font-medium rounded-2xl shadow-md hover:shadow-lg transition-all duration-200"
              >
                Create a Free Account
              </Button>
            </Link>
          </div>
          
          {/* Optional feature highlights */}
          <div className="flex flex-col sm:flex-row justify-center items-center gap-8 text-gray-600">
            <div className="flex items-center space-x-2">
              <BookOpen className="w-5 h-5 text-amber-600" />
              <span className="font-medium">One Place for Every Book</span>
            </div>
            <div className="flex items-center space-x-2">
              <Users className="w-5 h-5 text-amber-600" />
              <span className="font-medium">Thoughtful Tools, Not Feeds</span>
            </div>
            <div className="flex items-center space-x-2">
              <MessageCircle className="w-5 h-5 text-amber-600" />
              <span className="font-medium">Explore with Hypatia</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}