'use client'

import Link from 'next/link'
import { BookOpen } from 'lucide-react'

export default function Footer() {
  const leftLinks = [
    { name: 'About', href: '#about' },
    { name: 'Features', href: '#features' },
    { name: 'Contact', href: '#contact' },
    { name: 'Help', href: '/help' },
  ]

  const rightLinks = [
    { name: 'AI Ethics', href: '/ai-ethics' },
    { name: 'Creator Handbook', href: '/creator-handbook' },
    { name: 'Privacy', href: '/privacy' },
    { name: 'Terms', href: '/terms' },
  ]

  return (
    <footer className="bg-gray-900 text-white">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Footer Content */}
        <div className="py-12">
          {/* Brand Section */}
          <div className="text-center mb-8">
            <Link href="/" className="inline-flex items-center space-x-3 mb-6">
              <BookOpen className="h-8 w-8 text-amber-400" />
              <span className="text-2xl font-bold font-serif">Alexandria</span>
            </Link>
          </div>
          
          {/* Links */}
          <div className="flex flex-col md:flex-row justify-between items-center space-y-6 md:space-y-0">
            {/* Left Links */}
            <div className="flex flex-wrap justify-center md:justify-start gap-6">
              {leftLinks.map((link) => (
                <Link 
                  key={link.name}
                  href={link.href}
                  className="text-gray-300 hover:text-white transition-colors font-medium"
                >
                  {link.name}
                </Link>
              ))}
            </div>
            
            {/* Right Links */}
            <div className="flex flex-wrap justify-center md:justify-end gap-6">
              {rightLinks.map((link) => (
                <Link 
                  key={link.name}
                  href={link.href}
                  className="text-gray-300 hover:text-white transition-colors font-medium"
                >
                  {link.name}
                </Link>
              ))}
            </div>
          </div>
        </div>
        
        {/* Bottom Section */}
        <div className="border-t border-gray-800 py-8 text-center">
          <p className="text-gray-400 font-serif italic">
            Inspired by the ancient Library of Alexandria. Reimagined for the modern mind.
          </p>
          <div className="mt-4 text-gray-500 text-sm">
            © {new Date().getFullYear()} Alexandria. All rights reserved.
          </div>
        </div>
      </div>
    </footer>
  )
}