// =====================================================================
//  ALL THE TEXT ON THE SIMULATOR PAGE LIVES HERE. Edit freely.
// =====================================================================
//
//  How to edit
//  - Change anything between the quotes "..." or backticks `...`.
//    Backticks can span several lines; a blank line starts a new paragraph.
//  - Formatting you can use in most texts:
//      **bold**            [link text](https://example.org)
//      `code`              β_{τ}   (subscript: β with τ below)
//  - Words in {curly braces} are filled in by the page, e.g. {topic}, {pct}.
//    Keep the ones you want, delete the ones you don't. Each text lists
//    which ones it can use.
//  - Chart labels (marked "plain text") cannot use formatting.
//  - Save the file and reload the page to see your changes.
//
//  The explain menu (at the bottom) is a list: add, remove or reorder
//  entries as you like. The "What's this?" links point to entries by id.
// =====================================================================

window.TEXT = {

  // ---------- top of the page ----------
  page_title: "An interactive tool from the model of Candellone 2026",   // browser tab (plain text)
  header_title: "An interactive tool from the model of Candellone 2026",
  header_subtitle: "Try the model from my poster",
  intro: `Pick a debate topic and answer three questions. The model then predicts which side you would vote for, based on about 45,000 votes cast on debate.org between 2007 and 2022.`,

  // ---------- explain menu ----------
  explain_heading: "Explain",
  explain_placeholder: "Choose a question…",          // plain text

  // ---------- 1. set up ----------
  setup_heading: "1. Set up your vote",
  topic_label: "Debate topic",
  topic_option: "{Topic} ({n} debates)",              // plain text. {topic} {Topic} {n}

  opinion_label: "Your own opinion on the topic",
  opinion_symbol: "τ",
  opinion_options: ["Strongly Con", "Con", "None", "Pro", "Strongly Pro"],   // plain text, keep 5

  case_label: "Who made the better case?",
  case_symbol: "ρ",
  case_options: ["Con side", "Tie", "Pro side"],     // plain text, keep 3

  audience_label: "People who voted before you were…",
  audience_symbol: "η",
  audience_ends: ["alike", "typical", "very mixed"], // plain text, keep 3

  help_link: "What's this?",                          // plain text
  // Which explain entry each "What's this?" link opens. Use "" to hide a link.
  help_targets: { opinion: "opinion", case: "case", audience: "audience", arrows: "arrows" },

  examples_label: "Examples:",
  examples: {                                         // plain text
    conflict: "opinion vs. better case",
    strong: "strong opinion vs. better case",
    agree: "both agree",
    noview: "no opinion",
  },

  // ---------- 2. the prediction ----------
  result_heading: "2. The prediction",
  side_pro: "Pro",                                    // plain text
  side_con: "Con",
  // {side} {pct}
  result_big: "{side}, {pct}",
  // {side} {pct} {lo} {hi} {topic} {Topic} {rq1}
  result_sub: "Chance you vote for the {side} side (90% interval {lo}–{hi}). On {topic}, {rq1}.",

  // What the research finds for the selected topic; used as {rq1} above and in the topic list.
  rq1: {
    "persuasiveness-only": "only the better case matters reliably",
    "persuasiveness-driven": "the better case counts more than opinion",
    "jointly-driven": "both matter, no clear winner",
    "indeterminate": "neither is reliable (few debates)",
  },

  // One sentence under the prediction. Bold text is coloured: orange for the
  // better case, blue for opinion. {win} {lose} are the two arrow sizes.
  verdict_case_wins: "Your opinion and the better case point in different directions. Here **the better case counts for more** ({win} against {lose}).",
  verdict_opinion_wins: "Your opinion and the better case point in different directions. Here **your opinion counts for more** ({win} against {lose}).",
  verdict_agree: "Your opinion and the better case point the same way, so they add up.",
  verdict_only_case: "With no opinion, **only the better case** moves your vote.",
  verdict_only_opinion: "With a tie, **only your opinion** moves your vote.",
  verdict_none: "No opinion and a tie: the model has nothing to go on, so it stays close to 50/50.",

  arrows_caption: "Each arrow shows how far one answer moves the vote.",
  arrows_help_link: "How to read this",               // plain text

  // Arrow chart (plain text)
  row_opinion: "Your opinion",
  row_case: "Better case",
  row_audience: "Earlier voters",
  row_prediction: "Prediction",
  push_pro: "→ Pro",
  push_con: "← Con",
  push_none: "no push",
  prediction_pro: "{pct} Pro",                        // {pct}
  prediction_con: "{pct} Con",
  axis_ticks: ["95% Con", "75% Con", "50 / 50", "75% Pro", "95% Pro"],   // wide screens, keep 5
  axis_ticks_narrow: ["← Con", "50/50", "Pro →"],     // phones, keep 3
  // Tooltips when hovering an arrow (plain text)
  tip_opinion: "How far your own opinion moves the vote.",
  tip_case: "How far your judgement of the better case moves the vote.",
  tip_audience: "How far the mix of earlier voters moves the vote.",

  // Note under the arrows. {alike} {mixed} {topic} {Topic}
  audience_note_no_opinion: "Earlier voters can only change how much your opinion counts if you have one.",
  audience_note_reliable: "With alike vs. very mixed earlier voters, your opinion counts {alike} vs. {mixed}. {Topic} is one of only 4 topics where this difference is reliable.",
  audience_note_noise: "With alike vs. very mixed earlier voters, your opinion counts {alike} vs. {mixed}. On {topic} the model cannot tell this difference apart from zero.",

  // ---------- 3. all topics ----------
  all_heading: "3. The same vote on all 47 topics",
  all_summary_conflict: "With these answers, the better case counts for more than your opinion on {k} of 47 topics.",   // {k}
  all_summary_other: "With these answers, the predicted vote is Pro on {pro} of 47 topics.",   // {pro}
  all_caption: "Dot: predicted vote. Line: 90% interval. Tap a topic to choose it.",
  // Tooltip on a topic (plain text). {pct} {lo} {hi} {n} {rq1} {topic}
  all_tip_value: "{pct} Pro",
  all_tip_interval: "90% interval {lo}–{hi}",
  all_tip_meta: "{n} debates · {rq1}",

  // ---------- footer ----------
  footer: `**This page collects no data.** No cookies, no analytics, no tracking. Your answers are never sent anywhere: everything is calculated on your device and gone when you close the page.

Elena Candellone · Utrecht University · [paper](https://arxiv.org/abs/2606.03786) · [code](https://github.com/elenacandellone/DDO-conviction-conformity)`,

  // ---------- explain menu entries ----------
  // id: short name used by help_targets above (letters only)
  // question: shown in the drop-down (plain text)
  // answer: the explanation (formatting allowed, blank line = new paragraph)
  explain: [
    {
      id: "site",
      question: "What was debate.org?",
      answer: `debate.org was a website (2007–2022) where two people argued for and against a statement over several rounds. Anyone could then vote for a side.

Users could also list their own opinion on 48 political issues, such as abortion or gun rights, and votes were shown in the order they came in. That makes it possible to study opinions, arguments and audiences together.

The data were collected and released for research by Durmus and Cardie (2018, 2019).`,
    },
    {
      id: "opinion",
      question: "What does “your opinion” mean?",
      answer: `On debate.org, users could mark themselves Pro or Con on 48 issues. Here, “Pro” means you agree with the Pro side of the debate.

“Pro” and “Con” use a typical strength of opinion from the data (the median among users who gave one). “Strongly” uses the strongest 5%.

In the paper this is called conviction (τ).`,
    },
    {
      id: "case",
      question: "What does “better case” mean?",
      answer: `After reading a debate, voters said which debater made the more convincing arguments, separately from their vote. “Tie” means they picked neither.

This is the voter's own judgement, not an outside rating of the arguments. In the paper it is called persuasiveness (ρ).`,
    },
    {
      id: "audience",
      question: "What does “earlier voters” mean?",
      answer: `Votes came in one after another. For each vote, I measure how much the people who had already voted disagreed with each other across the 48 issues.

The slider runs from very alike earlier voters to very mixed ones (from the 5th to the 95th percentile of what happens in the data).

In the paper this is called cross-cutting exposure (η). The question is whether a mixed audience makes people rely less on their own opinion.`,
    },
    {
      id: "arrows",
      question: "How do I read the arrows?",
      answer: `Start from a voter with no opinion, a tie and a typical audience. That voter is close to 50/50.

Each arrow shows how far one of your answers moves the prediction: right towards Pro, left towards Con. The numbers are on the model's own scale (log-odds), where bigger means a stronger push.

The dot at the bottom is where you end up. The line around it is the 90% interval.`,
    },
    {
      id: "interval",
      question: "What is the 90% interval?",
      answer: `The model is Bayesian, so every effect comes with a range of plausible values instead of a single number. The 90% interval is the range the prediction falls in with 90% probability, given the data and the model.

Topics with many debates have narrow intervals. Topics with few debates have wide ones, so read those with care.

On this page the intervals are approximated: each effect's uncertainty is treated as a normal distribution.`,
    },
    {
      id: "strong",
      question: "Why can a strong opinion win?",
      answer: `Most users hold a moderate opinion. With a moderate opinion, the better case counts for more on nearly every topic.

Choose “Strongly” and a few topics flip. They are mostly topics with few debates and wide intervals, such as term limits or the gold standard.

The poster compares the two effects on the same scale (one standard deviation of each). On that comparison, own opinions never outweigh the better case on any of the 47 topics.`,
    },
    {
      id: "model",
      question: "What model is behind this?",
      answer: `A logistic regression with a separate effect for each of 47 topics, fitted in Stan:

\`logit P(vote Pro) = voter + debate + β_{τ}·opinion + β_{ρ}·better case + β_{η}·earlier voters + β_{τη}·opinion × earlier voters\`

The voter and debate terms absorb people who always lean one way and debates that attract one side. This page shows an average voter and an average debate.

Debate topics and stances were labelled with GPT-4o-mini and checked against a second classifier (SBERT + BART-MNLI).`,
    },
    {
      id: "findings",
      question: "What does the research find?",
      answer: `On 42 of 47 topics, the better case reliably predicts the vote. Own opinions do on 26. On no topic do own opinions outweigh the better case.

A more mixed audience changes how much opinions count on only 4 topics, and only one of those (animal rights) holds up with the second classifier.`,
    },
    {
      id: "limits",
      question: "What are the limits?",
      answer: `debate.org users chose to be there and were mostly from the US. They are not a representative sample of anyone.

Opinions are self-reported and may be out of date. “Better case” is the voter's own judgement, which is partly shaped by their opinion.

These are associations in observational data, not causal effects.`,
    },
    {
      id: "privacy",
      question: "Does this page collect my data?",
      answer: `No. This page has no cookies, no analytics and no tracking, and it does not store anything on your device. Your answers are never sent anywhere: all calculations run in your browser, and they are gone when you close the page.

The website hosting the page may keep ordinary server logs, as any website does. I do not use them.

The debate.org data are public research data. They are used only in aggregate, and no individual user is shown here.`,
    },
    {
      id: "who",
      question: "Who made this?",
      answer: `Elena Candellone, Utrecht University (Methodology & Statistics; Centre for Complex Systems Studies).

Paper: [arxiv.org/abs/2606.03786](https://arxiv.org/abs/2606.03786)

Code and models: [github.com/elenacandellone/DDO-conviction-conformity](https://github.com/elenacandellone/DDO-conviction-conformity)

Contact: e.candellone@uu.nl`,
    },
  ],
};
