"""
Interactive UI elements for bigcompute.science Colab notebooks.

Usage:
    from scripts.notebook_ui import celebrate, show_review, show_gpu_status
    celebrate("Zaremba Density", "A={1,2,8} at 10^10", {"density": "99.9999%", "uncovered": 500})
    show_review("kronecker-s30", "gemini-2.5-flash", "ACCEPT", "silver")
    show_gpu_status("Tesla T4", "16 GB", "100%")
"""

from IPython.display import HTML, display
import json
import random
import string


def _uid():
    return ''.join(random.choices(string.ascii_lowercase, k=8))


def celebrate(experiment_name, description, stats, gpu_name="Colab GPU"):
    """Render an animated celebration card with confetti after an experiment completes."""
    uid = _uid()
    stats_html = "".join(
        f'<div class="bc-stat"><span class="bc-stat-val">{v}</span><span class="bc-stat-key">{k}</span></div>'
        for k, v in stats.items()
    )

    html = f"""
    <div id="bc-{uid}" style="position:relative;overflow:hidden;background:#0b0d10;border:2px solid #e8c47a;border-radius:0;padding:0;margin:1rem 0;font-family:'Courier New',monospace;max-width:600px;">
      <canvas id="confetti-{uid}" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10;"></canvas>

      <div style="position:relative;z-index:1;padding:1.5rem 2rem;">
        <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;">
          <div style="width:8px;height:8px;background:#22c55e;border-radius:50%;box-shadow:0 0 8px #22c55e;"></div>
          <span style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.15em;color:#22c55e;font-weight:bold;">Computation Complete</span>
        </div>

        <h2 style="font-family:Georgia,serif;font-size:1.3rem;color:#e8c47a;margin:0 0 0.3rem;letter-spacing:-0.02em;">{experiment_name}</h2>
        <p style="font-size:0.8rem;color:#8a8580;margin:0 0 1.2rem;">{description}</p>

        <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:1.2rem;">
          {stats_html}
        </div>

        <div style="display:flex;align-items:center;gap:0.6rem;padding-top:0.8rem;border-top:1px solid #2a2e35;">
          <span style="font-size:0.6rem;color:#8a8580;">Computed on {gpu_name}</span>
          <span style="font-size:0.6rem;color:#2a2e35;">|</span>
          <span style="font-size:0.6rem;color:#e8c47a;">bigcompute.science</span>
          <span style="margin-left:auto;font-size:0.55rem;color:#8a8580;opacity:0.5;">You extended the frontier.</span>
        </div>
      </div>
    </div>

    <style>
      #bc-{uid} .bc-stat {{
        display:flex;flex-direction:column;
      }}
      #bc-{uid} .bc-stat-val {{
        font-family:'Courier New',monospace;font-size:1.4rem;font-weight:bold;color:#e8e6e3;line-height:1;
      }}
      #bc-{uid} .bc-stat-key {{
        font-size:0.55rem;text-transform:uppercase;letter-spacing:0.1em;color:#8a8580;margin-top:0.2rem;
      }}
    </style>

    <script>
    (function() {{
      const canvas = document.getElementById('confetti-{uid}');
      const ctx = canvas.getContext('2d');
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;

      const colors = ['#e8c47a', '#c41e1e', '#22c55e', '#5eead4', '#e85a1e', '#94a3b8'];
      const particles = [];

      for (let i = 0; i < 80; i++) {{
        particles.push({{
          x: Math.random() * canvas.width,
          y: -10 - Math.random() * canvas.height,
          w: 3 + Math.random() * 4,
          h: 6 + Math.random() * 8,
          color: colors[Math.floor(Math.random() * colors.length)],
          vy: 1 + Math.random() * 3,
          vx: (Math.random() - 0.5) * 2,
          rot: Math.random() * Math.PI * 2,
          vr: (Math.random() - 0.5) * 0.1,
          life: 1,
        }});
      }}

      let frame = 0;
      function draw() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        let alive = false;
        for (const p of particles) {{
          if (p.life <= 0) continue;
          alive = true;
          p.y += p.vy;
          p.x += p.vx;
          p.rot += p.vr;
          if (p.y > canvas.height * 0.7) p.life -= 0.02;

          ctx.save();
          ctx.translate(p.x, p.y);
          ctx.rotate(p.rot);
          ctx.globalAlpha = p.life;
          ctx.fillStyle = p.color;
          ctx.fillRect(-p.w/2, -p.h/2, p.w, p.h);
          ctx.restore();
        }}
        frame++;
        if (alive && frame < 200) requestAnimationFrame(draw);
      }}
      draw();
    }})();
    </script>
    """
    display(HTML(html))


def show_review(finding_slug, model, verdict, cert_level):
    """Render an animated review result card."""
    uid = _uid()

    verdict_color = {
        "ACCEPT": "#22c55e",
        "ACCEPT_WITH_REVISION": "#e8c47a",
        "REVISE_AND_RESUBMIT": "#c41e1e",
        "REJECT": "#ef4444",
    }.get(verdict, "#8a8580")

    cert_color = {
        "gold": "#e8c47a",
        "silver": "#94a3b8",
        "bronze": "#b45309",
    }.get(cert_level, "#8a8580")

    html = f"""
    <div id="rv-{uid}" style="background:#0b0d10;border-left:4px solid {verdict_color};padding:1.2rem 1.5rem;margin:0.8rem 0;max-width:600px;font-family:'Courier New',monospace;animation:rv-slide-{uid} 0.4s ease forwards;opacity:0;transform:translateX(-10px);">

      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
        <span style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;color:{verdict_color};font-weight:bold;">{verdict.replace('_', ' ')}</span>
        <span style="font-size:0.55rem;padding:0.15rem 0.5rem;background:{cert_color}22;color:{cert_color};border:1px solid {cert_color}44;text-transform:uppercase;letter-spacing:0.08em;font-weight:bold;">{cert_level}</span>
      </div>

      <h3 style="font-family:Georgia,serif;font-size:0.95rem;color:#e8e6e3;margin:0 0 0.3rem;">{finding_slug.replace('-', ' ').title()}</h3>
      <p style="font-size:0.7rem;color:#8a8580;margin:0;">Reviewed by <span style="color:#e8c47a;font-weight:bold;">{model}</span></p>
    </div>

    <style>
      @keyframes rv-slide-{uid} {{
        to {{ opacity: 1; transform: translateX(0); }}
      }}
    </style>
    """
    display(HTML(html))


def show_gpu_status(gpu_name, vram, utilization):
    """Render a GPU status card with animated utilization bar."""
    uid = _uid()
    util_num = int(utilization.replace('%', '')) if isinstance(utilization, str) else utilization
    bar_color = "#22c55e" if util_num < 50 else "#e8c47a" if util_num < 90 else "#c41e1e"

    html = f"""
    <div style="background:#0b0d10;border:1px solid #2a2e35;padding:1rem 1.5rem;margin:0.5rem 0;max-width:400px;font-family:'Courier New',monospace;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
        <span style="font-size:0.85rem;color:#e8e6e3;font-weight:bold;">{gpu_name}</span>
        <span style="font-size:0.65rem;color:#8a8580;">{vram} VRAM</span>
      </div>
      <div style="background:#1a1a1a;height:6px;width:100%;overflow:hidden;">
        <div id="bar-{uid}" style="height:100%;width:0%;background:{bar_color};transition:width 1s ease;"></div>
      </div>
      <span style="font-size:0.6rem;color:#8a8580;margin-top:0.3rem;display:block;">{utilization} utilization</span>
    </div>
    <script>
      setTimeout(() => document.getElementById('bar-{uid}').style.width = '{util_num}%', 100);
    </script>
    """
    display(HTML(html))


def show_leaderboard(entries):
    """Show a computation leaderboard.
    entries: list of dicts with 'name', 'value', 'unit', optional 'highlight'
    """
    rows = ""
    for i, e in enumerate(entries):
        is_highlight = e.get("highlight", False)
        bg = "#1a1710" if is_highlight else "transparent"
        border = "border-left:3px solid #e8c47a;" if is_highlight else "border-left:3px solid transparent;"
        badge = '<span style="font-size:0.5rem;background:#e8c47a;color:#0a0a0a;padding:0.1rem 0.4rem;font-weight:bold;margin-left:0.5rem;">YOU</span>' if is_highlight else ""
        rows += f"""
        <div style="display:flex;align-items:center;padding:0.5rem 0.8rem;{border}background:{bg};">
          <span style="font-size:0.7rem;color:#8a8580;width:1.5rem;">{i+1}.</span>
          <span style="font-size:0.78rem;color:#e8e6e3;flex:1;">{e['name']}{badge}</span>
          <span style="font-size:0.85rem;color:#e8c47a;font-weight:bold;font-family:'Courier New',monospace;">{e['value']}</span>
          <span style="font-size:0.55rem;color:#8a8580;margin-left:0.3rem;width:3rem;">{e.get('unit','')}</span>
        </div>
        """

    html = f"""
    <div style="background:#0b0d10;border:1px solid #2a2e35;max-width:500px;margin:1rem 0;font-family:'Courier New',monospace;overflow:hidden;">
      <div style="padding:0.6rem 0.8rem;border-bottom:1px solid #2a2e35;">
        <span style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;color:#e8c47a;font-weight:bold;">Computation Leaderboard</span>
      </div>
      {rows}
    </div>
    """
    display(HTML(html))
