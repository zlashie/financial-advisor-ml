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

public record Debt(
  @PositiveOrZero double amount,
  @DecimalMin(value="-0.5") @DecimalMax("1.0") double apr
) {}