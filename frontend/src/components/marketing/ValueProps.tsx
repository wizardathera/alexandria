'use client'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { BookOpen, Users, GraduationCap, ArrowRight } from 'lucide-react'

export default function ValueProps() {
  const accountTypes = [
    {
      icon: BookOpen,
      title: 'Readers',
      description: 'Build your personal library of understanding. Annotate deeply, ask Hypatia questions, and join or start discussions around the books that matter to you.',
      href: '#readers'
    },
    {
      icon: Users,
      title: 'Authors',
      description: 'Publish your ideas, annotate texts, and build followings around the conversations you care about.',
      href: '#authors'
    },
    {
      icon: GraduationCap,
      title: 'Educators',
      description: 'Curate learning experiences, guide discussions, and share insights with your students or community.',
      href: '#educators'
    }
  ]

  return (
    <>
      {/* Dare to Know Section */}
      <section className="py-20 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl sm:text-4xl font-bold font-serif text-gray-900 mb-8">
            Dare to Know
          </h2>
          <p className="text-xl text-gray-700 leading-relaxed font-serif">
            Alexandria is a human-centered reading platform that helps you think deeply, 
            learn intentionally, and connect through books.
          </p>
        </div>
      </section>

      {/* Choose Your Path Section */}
      <section className="py-20 bg-slate-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold font-serif text-gray-900 mb-4">
              Choose Your Path
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {accountTypes.map((type) => (
              <Card 
                key={type.title}
                className="bg-white border border-gray-200 rounded-2xl hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
              >
                <CardHeader className="text-center pb-4">
                  <div className="w-12 h-12 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <type.icon className="w-6 h-6 text-amber-700" />
                  </div>
                  <CardTitle className="text-xl font-bold font-serif text-gray-900">
                    {type.title}
                  </CardTitle>
                </CardHeader>
                
                <CardContent className="text-center">
                  <CardDescription className="text-gray-600 mb-6 leading-relaxed font-serif">
                    {type.description}
                  </CardDescription>
                  
                  <Button 
                    variant="ghost"
                    className="text-indigo-700 hover:text-indigo-800 font-medium p-0"
                  >
                    Learn more →
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>
    </>
  )
}