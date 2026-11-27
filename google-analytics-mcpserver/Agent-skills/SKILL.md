# Google GMA Next-Gen SDK

[Agent skills](https://antigravity.google/docs/skills) for GMA Next-Gen SDK are
portable, self-contained modules of GMA Next-Gen SDK knowledge, instructions, and
workflows. These skills are designed to help AI assistants understand
GMA Next-Gen SDK and run complex tasks with higher accuracy. For all available
agent skills, see the [`ads`](https://github.com/google/skills/tree/main/skills/ads/)
folder in the GitHub repository.

This page covers installing and updating agent skills.

## Install the agent skills

Agent skills for GMA Next-Gen SDK work with AI assistants that support skills,
including Antigravity, Claude Code, Codex, and Cursor.

To install agent skills for GMA Next-Gen SDK, run the following command:

    npx skills add google/skills/skills/ads

Agent skills for GMA Next-Gen SDK begin with `google-mobile-ads`.

## Update the agent skills

To make sure AI assistant has the latest GMA Next-Gen SDK features and
recommendations, update your agent skills. The following example updates all
agent skills for GMA Next-Gen SDK:

    npx skills update --all

## View available skills

You can use agent skills for GMA Next-Gen SDK by prompting your preferred AI
assistant. The following table describes all available skills and sample
prompts:

| Skill | Description | Sample prompt |
|---|---|---|
| `google-mobile-ads-get-started` | Assists with adding the GMA Next-Gen SDK to your app for the first time, setting the application ID, and initializing GMA Next-Gen SDK. | *Help me initialize the GMA Next-Gen SDK* |
| `google-mobile-ads-banner` | Assists with implementing banner ads, such as large anchored adaptive banners and inline adaptive banners. | *Add a banner ad to my app* |
| `google-mobile-ads-interstitial` | Assists with implementing full-screen interstitial ads and showing the ads at natural transition points in your app. | *Add an interstitial ad to my app* |
| `google-mobile-ads-rewarded` | Assists with implementing rewarded ads, showing the ads, and handling user reward callbacks. | *Implement a rewarded ad* |
| `google-mobile-ads-android-migrate-to-next-gen` | Assists with migrating an existing Android app from Google Mobile Ads SDK (Legacy) to GMA Next-Gen SDK. | *Migrate my app to GMA Next-Gen SDK* |
| `google-mobile-ads-validate` | Assists with validating a project's GMA Next-Gen SDK integration, conducting pre-launch audits, and checking against common implementation errors. | *Validate my GMA Next-Gen SDK integration* |
