import { useEffect, useRef, useState } from 'react'

// Animates a number from 0 → target with an ease-out curve.
// Honors prefers-reduced-motion by jumping straight to the target.
export function useCountUp(target = 0, duration = 1200) {
  const [value, setValue] = useState(0)
  const frame = useRef(null)

  useEffect(() => {
    const reduce =
      typeof window !== 'undefined' &&
      window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches

    if (reduce || !target) {
      setValue(target)
      return
    }

    const start = performance.now()
    const tick = (now) => {
      const t = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - t, 3) // easeOutCubic
      setValue(Math.round(target * eased))
      if (t < 1) frame.current = requestAnimationFrame(tick)
    }
    frame.current = requestAnimationFrame(tick)

    return () => {
      if (frame.current) cancelAnimationFrame(frame.current)
    }
  }, [target, duration])

  return value
}
