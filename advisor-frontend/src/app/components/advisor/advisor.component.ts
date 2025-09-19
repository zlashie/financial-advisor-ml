import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { NgFor, NgIf } from '@angular/common';
import {
  RecommendationService,
  Debt,
  Investments,
  RecommendationRequest
} from '../../services/recommendation.service';

@Component({
  selector: 'app-advisor',
  standalone: true,
  imports: [FormsModule, NgFor, NgIf],
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

  constructor(private recommendationService: RecommendationService) {}

  addDebt() { this.debts.push({ amount: 0, apr: 0 }); }

  calculate() {
    const payload: RecommendationRequest = {
      debts: this.debts,
      investments: this.investments,
      horizon_years: this.horizon_years,
      monthly_extra: this.monthly_extra
    };
    this.loading = true;
    this.recommendationService.getRecommendation(payload).subscribe({
      next: res => this.result = res,
      error: err => console.error('Error fetching recommendation', err),
      complete: () => this.loading = false
    });
  }
}
