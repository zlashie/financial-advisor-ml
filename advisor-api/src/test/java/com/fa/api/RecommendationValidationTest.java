/*
************************
* Dependencies
************************
*/

package com.fa.api;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.reactive.server.WebTestClient;

import java.io.IOException;

/*
************************
* Test
************************
*/

@SpringBootTest
@AutoConfigureWebTestClient
class RecommendationValidationTest {

  static MockWebServer ml = new MockWebServer();

  @DynamicPropertySource
  static void props(DynamicPropertyRegistry r) {
    // point the API to our mock ML for the VALID payload test
    r.add("ml.base-url", () -> "http://localhost:" + ml.getPort());
  }

  @BeforeAll static void start() throws IOException { ml.start(); }
  @AfterAll static void stop() throws IOException { ml.shutdown(); }

  @Autowired WebTestClient client;

  String goodReq = """
  {"debts":[{"amount":12000.0,"apr":0.199}],
   "investments":{"equity_value":15000.0,"property_value":300000.0,"property_growth_rate":0.02,"equity_return_rate":0.08},
   "horizon_years":10,"monthly_extra":1000.0}
  """;

  @Test
  void validPayload_returns_200_when_ml_ok() {
    var mlResponse = """
    {"allocation":{"ratio":0.65,"to_debt":650.0,"to_investing":350.0},
     "projections":{"years":[2025,2026],"assets_pos":[100.0,110.0],
                    "debt_neg":[-50.0,-20.0],"net_worth":[50.0,90.0]},
     "explain":["ok"]}
    """;
    ml.enqueue(new MockResponse().setHeader("Content-Type","application/json").setBody(mlResponse));

    client.post().uri("/api/recommendation")
      .contentType(MediaType.APPLICATION_JSON)
      .bodyValue(goodReq)
      .exchange()
      .expectStatus().isOk()
      .expectBody()
      .jsonPath("$.allocation.ratio").isEqualTo(0.65);
  }

  @Test
  void apr_out_of_range_returns_400() {
    var bad = """
    {"debts":[{"amount":12000.0,"apr":1.5}],
     "investments":{"equity_value":0,"property_value":0,"property_growth_rate":0.02,"equity_return_rate":0.08},
     "horizon_years":10,"monthly_extra":1000.0}
    """;
    client.post().uri("/api/recommendation")
      .contentType(MediaType.APPLICATION_JSON)
      .bodyValue(bad)
      .exchange()
      .expectStatus().isBadRequest();
  }

  @Test
  void horizon_below_min_returns_400() {
    var bad = """
    {"debts":[{"amount":12000.0,"apr":0.199}],
     "investments":{"equity_value":0,"property_value":0,"property_growth_rate":0.02,"equity_return_rate":0.08},
     "horizon_years":0,"monthly_extra":1000.0}
    """;
    client.post().uri("/api/recommendation")
      .contentType(MediaType.APPLICATION_JSON)
      .bodyValue(bad)
      .exchange()
      .expectStatus().isBadRequest();
  }

  @Test
  void negative_equity_value_returns_400() {
    var bad = """
    {"debts":[{"amount":12000.0,"apr":0.199}],
     "investments":{"equity_value":-1,"property_value":0,"property_growth_rate":0.02,"equity_return_rate":0.08},
     "horizon_years":10,"monthly_extra":1000.0}
    """;
    client.post().uri("/api/recommendation")
      .contentType(MediaType.APPLICATION_JSON)
      .bodyValue(bad)
      .exchange()
      .expectStatus().isBadRequest();
  }
}