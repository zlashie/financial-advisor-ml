/*
************************
* Dependencies
************************
*/

package com.fa.api.dto;

import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import java.util.List;

/*
************************
* Main
************************
*/

public record RecommendationRequest(
  @NotNull List<@Valid Debt> debts,   
  @NotNull @Valid Investments investments,  
  @Min(1) @Max(60) int horizon_years,
  @PositiveOrZero double monthly_extra
) {}