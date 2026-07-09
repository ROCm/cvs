import { Clock } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/Card'

export function ComingSoon({ title }: { title: string }) {
  return (
    <Card>
      <CardContent className="py-16 text-center text-gray-400">
        <Clock className="h-12 w-12 mx-auto mb-4 opacity-40" />
        <p className="text-lg font-medium text-gray-500">{title}</p>
        <p className="text-sm mt-1">This section is coming soon.</p>
      </CardContent>
    </Card>
  )
}
