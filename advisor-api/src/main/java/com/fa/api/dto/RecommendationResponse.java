/*
************************
* Dependencies
************************
*/

package com.fa.api.dto;
import java.util.List;

/*
************************
* Main
************************
*/

public record RecommendationResponse(
    AllocationResponse allocation,
    ProjectionResponse projections,
    List<String> explain
) {}

record AllocationResponse(
    double ratio,
    double to_debt,
    double to_investing
) {}

record ProjectionResponse(
    List<Integer> years,
    List<Double> assets_pos,
    List<Double> debt_neg,
    List<Double> net_worth
) {}