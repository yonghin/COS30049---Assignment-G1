import * as d3 from 'd3'
import { getThemeColors } from './chartTheme'

// Download the SVG in container as a PNG (2× for crisp HiDPI output).
export function downloadPNG(container, filename = 'chart') {
  const svgNode = container.querySelector('svg')
  if (!svgNode) return

  const { bg } = getThemeColors()
  const vb = svgNode.viewBox.baseVal
  const W  = vb.width  || svgNode.clientWidth  || 600
  const H  = vb.height || svgNode.clientHeight || 400
  const scale = 2

  const clone = svgNode.cloneNode(true)
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  // Reset viewBox so the clone captures the full chart, not a zoomed sub-region
  clone.setAttribute('viewBox', `0 0 ${W} ${H}`)
  clone.style.background = bg

  const svgStr = new XMLSerializer().serializeToString(clone)
  const blob   = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' })
  const url    = URL.createObjectURL(blob)

  const img = new Image()
  img.onload = () => {
    const canvas = document.createElement('canvas')
    canvas.width  = W * scale
    canvas.height = H * scale
    const ctx = canvas.getContext('2d')
    ctx.scale(scale, scale)
    ctx.fillStyle = bg
    ctx.fillRect(0, 0, W, H)
    ctx.drawImage(img, 0, 0)
    URL.revokeObjectURL(url)

    canvas.toBlob(pngBlob => {
      const a    = document.createElement('a')
      a.download = `${filename}.png`
      a.href     = URL.createObjectURL(pngBlob)
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      setTimeout(() => URL.revokeObjectURL(a.href), 1000)
    }, 'image/png')
  }
  img.onerror = () => URL.revokeObjectURL(url)
  img.src = url
}

// Add a toolbar with optional +/−/Reset buttons and always a Save (PNG) button.
//   svgSel      — d3 selection of the element that zoomBehavior is applied to
//   zoomBehavior — d3.zoom() instance
//   onReset     — extra callback fired when Reset is clicked (e.g. reset zoomRef)
//   title       — filename used for PNG download
//   hasPanZoom  — false = Save button only (e.g. DonutChart)
export function addToolbar(container, {
  svgSel       = null,
  zoomBehavior = null,
  onReset      = null,
  title        = 'chart',
  hasPanZoom   = true,
} = {}) {
  const toolbar = d3.select(container)
    .append('div')
    .attr('class', 'chart-toolbar')
    .style('position', 'absolute')
    .style('top', '6px')
    .style('right', '8px')
    .style('display', 'flex')
    .style('gap', '4px')
    .style('z-index', '10')

  const mkBtn = (label, color, onClick) =>
    toolbar.append('button')
      .style('background', 'transparent')
      .style('color', color)
      .style('border', `1.5px solid ${color}`)
      .style('border-radius', '4px')
      .style('padding', '2px 8px')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .style('cursor', 'pointer')
      .style('line-height', '1.6')
      .style('transition', 'opacity 0.15s')
      .text(label)
      .on('mouseover', function () { d3.select(this).style('opacity', '0.65') })
      .on('mouseout',  function () { d3.select(this).style('opacity', '1') })
      .on('click', onClick)

  if (hasPanZoom && svgSel && zoomBehavior) {
    mkBtn('+', '#00d4ff',
      () => svgSel.transition().duration(250).call(zoomBehavior.scaleBy, 1.5))
    mkBtn('−', '#00d4ff',
      () => svgSel.transition().duration(250).call(zoomBehavior.scaleBy, 1 / 1.5))
    mkBtn('⟳', '#f59e0b', () => {
      if (onReset) onReset()
      svgSel.transition().duration(300).call(zoomBehavior.transform, d3.zoomIdentity)
    })
  }

  mkBtn('⬇ Save', '#10b981', () => downloadPNG(container, title))
}

// Convenience: create a viewBox-based zoom behavior for charts without axis rescaling.
// The zoom handler updates the SVG viewBox instead of repositioning elements.
// Returns { zoomBehavior } — call svg.call(zoomBehavior) after invoking this.
export function makeViewBoxZoom(svg, W, H, zoomTransformRef) {
  const origViewBox = `0 0 ${W} ${H}`

  const zoomBehavior = d3.zoom()
    .filter(e => e.type !== 'wheel')   // buttons only — no scroll-wheel zoom
    .scaleExtent([0.5, 10])
    .on('zoom', event => {
      zoomTransformRef.current = event.transform
      const t  = event.transform
      const vW = W / t.k
      const vH = H / t.k
      const vX = -t.x / t.k
      const vY = -t.y / t.k
      svg.attr('viewBox', `${vX} ${vY} ${vW} ${vH}`)
    })

  svg.call(zoomBehavior).call(zoomBehavior.transform, zoomTransformRef.current)

  return {
    zoomBehavior,
    resetViewBox: () => svg.attr('viewBox', origViewBox),
  }
}
