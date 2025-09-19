import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AdvisorComponent } from './app/components/advisor/advisor.component';

bootstrapApplication(AdvisorComponent, appConfig)
  .catch(err => console.error(err));
