'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/store/auth-store'
import Header from '@/components/marketing/Header'
import Hero from '@/components/marketing/Hero'
import ValueProps from '@/components/marketing/ValueProps'
import HypatiaIntro from '@/components/marketing/HypatiaIntro'
import HowItWorks from '@/components/marketing/HowItWorks'
import CreatorTools from '@/components/marketing/CreatorTools'
import FinalCTA from '@/components/marketing/FinalCTA'
import Footer from '@/components/marketing/Footer'

export default function HomePage() {
  const router = useRouter()
  const { isAuthenticated } = useAuthStore()

  useEffect(() => {
    if (isAuthenticated) {
      router.push('/library')
    }
  }, [isAuthenticated, router])

  // If user is authenticated, redirect to library (handled by useEffect)
  if (isAuthenticated) {
    return null
  }

  return (
    <div className="min-h-screen bg-white">
      <Header />
      <main>
        {/* Section: Hero */}
        <Hero />
        
        {/* Section: Value Propositions */}
        <ValueProps />
        
        {/* Section: Hypatia Introduction */}
        <HypatiaIntro />
        
        {/* Section: How It Works */}
        <HowItWorks />
        
        {/* Section: Creator Tools */}
        <CreatorTools />
        
        {/* Section: Final CTA */}
        <FinalCTA />
      </main>
      <Footer />
    </div>
  )
}