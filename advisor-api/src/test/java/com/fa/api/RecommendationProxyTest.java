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
class RecommendationProxyTest {

  static MockWebServer ml = new MockWebServer();

  @DynamicPropertySource
  static void overrideProps(DynamicPropertyRegistry r) {
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
  void validPayload_isProxiedToMl_andReturned() {
    String mlResponse = """
    {"allocation":{"ratio":0.65,"to_debt":650.0,"to_investing":350.0},
     "projections":{"years":[2025,2026],"assets_pos":[100.0,110.0],
                    "debt_neg":[-50.0,-20.0],"net_worth":[50.0,90.0]},
     "explain":["ok"]}
    """;
    ml.enqueue(new MockResponse().setBody(mlResponse).setHeader("Content-Type", "application/json"));

    client.post().uri("/api/recommendation")
      .contentType(MediaType.APPLICATION_JSON)
      .bodyValue(goodReq)
      .exchange()
      .expectStatus().isOk()
      .expectBody()
      .jsonPath("$.allocation.ratio").isEqualTo(0.65)
      .jsonPath("$.projections.years.length()").isEqualTo(2)
      .jsonPath("$.explain[0]").isEqualTo("ok");
  }
}