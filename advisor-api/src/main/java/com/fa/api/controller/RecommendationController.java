/*
************************
* Dependencies
************************
*/

package com.fa.api.controller;

import com.fa.api.dto.*;
import jakarta.validation.Valid;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

/*
************************
* Main
************************
*/

@RestController
@RequestMapping("/api")
public class RecommendationController {

  private final WebClient mlClient;

  public RecommendationController(WebClient mlClient) {
    this.mlClient = mlClient;
  }

  @PostMapping(
    value = "/recommendation",
    consumes = MediaType.APPLICATION_JSON_VALUE,
    produces = MediaType.APPLICATION_JSON_VALUE
  )
  public Mono<RecommendationResponse> recommend(@Valid @RequestBody RecommendationRequest req) {
    // Forward payload to ML service
    return mlClient.post()
      .uri("/inference/recommend")
      .contentType(MediaType.APPLICATION_JSON)
      .bodyValue(req)
      .retrieve()
      .bodyToMono(RecommendationResponse.class);
  }
}