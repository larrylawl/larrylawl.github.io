---

## title: 'Lessons on Personalization'

date: 2026-07-01
permalink: /posts/2026/07/perso-lessons
tags:

- data science

Having done about 1.5 years of personalization across multiple clients, I wanted to consolidate my takeaways here. Personalization is here to stay because of it's contextual nature; hopefully someone doing perso will find them useful!

## GCG is hard to beat

No clients I worked with beat global control group (GCG). The fact that this was consistent across clients underscored that GCG is truly hard to beat. This can be counter intuitive - how can doing something be possibly worse than doing nothing? The most common reasons I've seen are: 

1. BTL offers - giving below-the-line (BTL) offers to false positive customers (e.g. customers who would have anyway bought the product / would not have churned). We need to be extra careful in giving BTL offers.
2. Awaken sleeping customers - by proactively reaching out to customers, customers may realize that their current deal is not a good deal and thus downgrade / churn.

And by setting up GCG, you can slice and dice to find out campaigns which are not working and cut those. Easy do-nothing interventions that lead to incredible revenue gains. Very effort efficient.

## Importance of uplift modelling

> Primer on the different modelling objectives that get increasingly close to the business objective. Credits: Julian King


| **Approach**                      | **Key question**                                                                                       | **Theoretical problem** |
| --------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------- |
| **Propensity modelling**          | *"What is the probability that a customer will do Y?"*                                                 | **P(churn               |
| **Response modelling**            | *"What is the probability that a customer will do Y, if we contact them?"*                             | **P(churn               |
| **Contextual response modelling** | *"What is the probability that a customer will do Y, if we contact them and talk about Z?"*            | **P(churn               |
| **Uplift modelling**              | *"What is the incremental probability that a customer will do Y if we contact them and talk about Z?"* | **P(churn               |


Because of how strong GCG is as a baseline, uplift modelling becomes super important because we account for GCG as part of the modelling objective. Uplift modelling accounts for the probably of churn given not contacted, which is exactly the behaviour of GCG. In my previous case, not accounting for GCG had material revenue issues: we ended up awakening sleeping customers who realized that their current deal is not a good one and thus downgraded. In this case (which is a downgrade), it's important to account for both probability of take up AND revenue. To account for both, the cleanest way is to calculate expected revenue (which is prob * revenue). But in practice, this becomes complicated as it requires having two models (classifier + regressor). I suggest to start with the mathematically sound prob * revenue, before trying something funky like top 3 deciles of classifier going into regressor. Ah, I digress.

## Setting up GCG should be top priority

There's SO much value in setting up GCG as to priority. It sets up your measurement baseline. You can stop dilutive campaigns and see immediate lift (thus boost confidence). Then you start collecting data for uplift modelling. Incredible value.

## Instead of focusing on churn, focus on the save-a-customer process

Churn often gets alot of attention because of it's significant revenue impact (CLTV, market share etc). But we often overcompensate the treatment and end up having way too many false positives (ie people who would not have churned but identified as churners), thus having ARPU dilution. Simulation needs to be done to check that saving the true positives outweigh the ARPU dilution of false positives. From what i've seen, it's hard to do proactive churn. Instead, focus on the **save-a-customer process**. By definition, these are customers who HAVE indicated their intention to churn. 100% true positives. So focus on the process that saves a customer. Examples of things that worked included: BTL offers, comparison against customers.

## GenAI for outbound sales is a tough nut to crack

With the GenAI hype, multiple projects have tried to use genAI for outbound sales. From what i've seen, there's potential but it's heavily limited by how expensive whatsapp costs are. At time of writing, whatsapp cost 25cents per message, with the cost reimbursed if the customer replied. The very structure is not favourable for genAI outbound sales, which are meant to be mass sent with lower conversion fate. Moreover, the (lack of) follow up interaction point begs the question - do we really need genAI? Or can the first message blast suffice?

This being said, it was still amazing to witness a fully automnomous agent make a sale. As Andrew Ng said, we should bet on how / where the technology is trending, not where it is currently

## Some other cool tidbits

- **Gap window**. Do include a gap window between the inference snapshot and the predicted action. The reason for the gap window is not only for leakage but also to take into account the operional delay between data generation and when the action reaches a customer. It's not realistic that you can contact someone on the first of May with predictions based on April data. So it's not helpful to assume zero delay and count these kind of events into the performance and optimize the model for it. You might end up training your model to predict based on features that dont work in practise.
- **Ensure dev infra is up before DS work starts**. Dev infra includes: proper tooling for software dev (codex, cursor, VMs), connected to client DB and to cloud services. A common mistake of consulting projects is that all teams start together when project launches. But we often need engineering team to set up the dev infra properly before the DS team becomes efficient.
- **Perso starter codebase**. I wrote a production-ready starter repository for tabular machine learning, designed to be cloud- and database-agnostic. I hope it'll be useful for another project elsewhere! Link [here.](https://github.com/larrylawl/tabular-ml-boilerplate) Do give the repo a star if you found it useful.

Hope this helps someone doing perso out there!

## Credits

Full credits go to the BCG DeepAI team. In particular to [Thomas](https://www.linkedin.com/in/thomasfsbuettner/), who I've learnt so much from. 