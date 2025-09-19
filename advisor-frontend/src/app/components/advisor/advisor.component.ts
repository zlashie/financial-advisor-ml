import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common'; 
import { RecommendationService, Debt, Investments, RecommendationRequest } from '../../services/recommendation.service';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartOptions, ChartType, Plugin } from 'chart.js';
import { Chart, registerables } from 'chart.js';
import { ViewChild } from '@angular/core';
Chart.register(...registerables);

@Component({
  selector: 'app-root',                
  standalone: true,
  imports: [CommonModule, FormsModule, BaseChartDirective],
  templateUrl: './advisor.component.html',
  styleUrls: ['./advisor.component.css']
})
export class AdvisorComponent {
  @ViewChild(BaseChartDirective) chart?: BaseChartDirective;

  // ---- form model ----
  debts: Debt[] = [{ amount: 0, apr: 4 }];
  investments: Investments = {
    equity_value: 0,
    property_value: 0,
    property_growth_rate: 4,
    equity_return_rate: 8
  };
  horizon_years = 20;
  monthly_extra = 1000;

  // ---- state ----
  result: any = null;
  loading = false;
  error: string | null = null;

  // ---- chart state ----
  lineChartType: ChartType = 'line';
  lineChartData: ChartConfiguration['data'] = {
    labels: [],
    datasets: [
      { label: 'Assets (+)', data: [], borderWidth: 2, pointRadius: 0, borderColor: '#2e7d32', fill: false, tension: 0.1 },
      { label: 'Debt (−)',   data: [], borderWidth: 2, pointRadius: 0, borderColor: '#c62828', fill: false, tension: 0.1 },
      { label: 'Net Worth',  data: [], borderWidth: 2, pointRadius: 0, borderColor: '#455a64', fill: false, borderDash: [6,4], tension: 0.1 }
    ]
  };

  zeroLinePlugin: Plugin = {
    id: 'zeroLine',
    afterDraw: (chart) => {
      const yScale: any = chart.scales['y'];
      if (!yScale) return;
      const y = yScale.getPixelForValue(0);
      const ctx = chart.ctx as CanvasRenderingContext2D;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(chart.chartArea.left, y);
      ctx.lineTo(chart.chartArea.right, y);
      ctx.lineWidth = 1;
      ctx.strokeStyle = '#9e9e9e';
      ctx.setLineDash([2,3]);
      ctx.stroke();
      ctx.restore();
    }
  };

  lineChartOptions: ChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'bottom' },
      tooltip: { enabled: true }
    },
    scales: {
      x: { title: { display: true, text: 'Year' } },
      y: { title: { display: true, text: 'Amount (net worth baseline at 0)' } }
    }
  };

  constructor(private svc: RecommendationService) {}

  addDebt() { this.debts.push({ amount: 0, apr: 0 }); }
  removeDebt(i: number) { if (this.debts.length > 1) this.debts.splice(i, 1); }

  private normalizeRate(r: number): number {
    if (r == null || isNaN(r as any)) return 0;
    return r > 1 ? r / 100 : r;
  }

  calculate() {
    const debtsNorm = this.debts.map(d => ({
      amount: Number(d.amount) || 0,
      apr: this.normalizeRate(Number(d.apr))
    }));

    const inv = {
      equity_value: Number(this.investments.equity_value) || 0,
      property_value: Number(this.investments.property_value) || 0,
      property_growth_rate: this.normalizeRate(Number(this.investments.property_growth_rate)),
      equity_return_rate: this.normalizeRate(Number(this.investments.equity_return_rate))
    };

    const anyBadApr = debtsNorm.some(d => d.apr < -0.5 || d.apr > 1.0);
    const badEquity = inv.equity_return_rate < -0.5 || inv.equity_return_rate > 1.0;
    const badProp   = inv.property_growth_rate < -0.5 || inv.property_growth_rate > 1.0;
    const badHorizon = this.horizon_years < 1 || this.horizon_years > 60;

    if (anyBadApr || badEquity || badProp || badHorizon) {
      this.error = 'One or more inputs are out of range. Use 0–1 for rates (or 0–100%, we convert). Horizon 1–60.';
      return;
    }

    const payload: RecommendationRequest = {
      debts: debtsNorm,
      investments: inv,
      horizon_years: Number(this.horizon_years),
      monthly_extra: Number(this.monthly_extra) || 0
    };

    this.loading = true;
    this.error = null;

    this.svc.getRecommendation(payload).subscribe({
      next: (res) => {
        this.result = res;

        const years = res.projections.years;
        const assets = res.projections.assets_pos;
        const debt = res.projections.debt_neg; 
        const net = res.projections.net_worth ?? assets.map((a: number, i: number) => a + debt[i]);

        this.lineChartData = {
          labels: years,
          datasets: [
            { ...this.lineChartData.datasets[0], data: assets },
            { ...this.lineChartData.datasets[1], data: debt },
            { ...this.lineChartData.datasets[2], data: net }
          ]
        };
        this.chart?.update();
      },
      error: () => this.error = 'Request failed',
      complete: () => this.loading = false
    });
  }
}
