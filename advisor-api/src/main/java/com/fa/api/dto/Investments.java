/*
************************
* Dependencies
************************
*/

package com.fa.api.dto;
import jakarta.validation.constraints.*;

/*
************************
* Main
************************
*/


public record Investments(
  @PositiveOrZero double equity_value,
  @PositiveOrZero double property_value,
  @DecimalMin(value="-0.5") @DecimalMax("1.0") double property_growth_rate,
  @DecimalMin(value="-0.5") @DecimalMax("1.0") double equity_return_rate
) {}
