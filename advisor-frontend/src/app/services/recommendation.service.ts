// src/app/services/recommendation.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Debt { amount: number; apr: number; }
export interface Investments {
  equity_value: number;
  property_value: number;
  property_growth_rate: number;
  equity_return_rate: number;
}
export interface RecommendationRequest {
  debts: Debt[];
  investments: Investments;
  horizon_years: number;
  monthly_extra: number;
}

@Injectable({ providedIn: 'root' })
export class RecommendationService {
  private apiUrl = '/api/recommendation';

  constructor(private http: HttpClient) {}

  getRecommendation(req: RecommendationRequest): Observable<any> {
    return this.http.post<any>(this.apiUrl, req);
  }
}
