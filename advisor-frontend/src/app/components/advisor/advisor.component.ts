import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { NgFor, NgIf, DecimalPipe, JsonPipe } from '@angular/common';  // ⟵ add DecimalPipe, JsonPipe
import {
  RecommendationService,
  Debt,
  Investments,
  RecommendationRequest
} from '../../services/recommendation.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    FormsModule,
    NgFor, NgIf,
    DecimalPipe, JsonPipe    
  ],
  templateUrl: './advisor.component.html',
  styleUrls: ['./advisor.component.css']
})
export class AdvisorComponent {
  debts: Debt[] = [{ amount: 0, apr: 0 }];
  investments: Investments = {
    equity_value: 0,
    property_value: 0,
    property_growth_rate: 0.02,
    equity_return_rate: 0.08
  };
  horizon_years = 10;
  monthly_extra = 1000;

  result: any = null;
  loading = false;
  error: string | null = null;

  constructor(private recommendationService: RecommendationService) {}

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
    this.recommendationService.getRecommendation(payload).subscribe({
      next: res => this.result = res,
      error: err => { console.error(err); this.error = 'Request failed'; },
      complete: () => this.loading = false
    });
  }
}
