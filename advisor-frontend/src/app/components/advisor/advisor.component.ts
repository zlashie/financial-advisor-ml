import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common'; 
import { RecommendationService, Debt, Investments, RecommendationRequest } from '../../services/recommendation.service';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartOptions, ChartType, Plugin } from 'chart.js';
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

@Component({
  selector: 'app-root',                
  standalone: true,
  imports: [CommonModule, FormsModule, BaseChartDirective],
  templateUrl: './advisor.component.html',
  styleUrls: ['./advisor.component.css']
})
export class AdvisorComponent {
  // ---- form model ----
  debts: Debt[] = [{ amount: 0, apr: 0 }];
  investments: Investments = {
    equity_value: 0,
    property_value: 0,
    property_growth_rate: 0.02,
    equity_return_rate: 0.08
  };
  horizon_years = 10;
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

  calculate() {
    const payload: RecommendationRequest = {
      debts: this.debts,
      investments: this.investments,
      horizon_years: this.horizon_years,
      monthly_extra: this.monthly_extra
    };
    this.loading = true;
    this.error = null;
    this.svc.getRecommendation(payload).subscribe({
      next: (res) => {
        this.result = res;
        const years = res.projections.years;
        const assets = res.projections.assets_pos;
        const debt = res.projections.debt_neg;        
        const net = res.projections.net_worth ?? assets.map((a: number, idx: number) => a + debt[idx]);

        this.lineChartData = {
          labels: years,
          datasets: [
            { ...this.lineChartData.datasets[0], data: assets },
            { ...this.lineChartData.datasets[1], data: debt },
            { ...this.lineChartData.datasets[2], data: net }
          ]
        };
      },
      error: () => this.error = 'Request failed',
      complete: () => this.loading = false
    });
  }
}
