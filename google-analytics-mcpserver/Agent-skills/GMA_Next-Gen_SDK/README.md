# Integrating GMA Next-Gen SDK 

[Video](https://www.youtube.com/watch?v=giwn21KeYoQ)

Integrating GMA Next-Gen SDK into an app is the first step toward displaying ads
and earning revenue. Once you've integrated the SDK, you can choose an ad
format (such as native or rewarded video) and follow the steps to implement it.

## Before you begin

To prepare your app, complete the steps in the following sections.

### App prerequisites

- Make sure that your app's build file uses the following values:

  - Minimum SDK version of `24` or higher
  - Compile SDK version of `35` or higher

<!-- -->

- For Kotlin apps, use the minimum Kotlin version 1.9.

### Set up your app in your AdMob account

Register your app as an AdMob app by completing the following steps:

1. [Sign in to](https://admob.google.com/home/) or [sign up
   for](https://support.google.com/admob/answer/7356219) an AdMob account.

2. [Register your app with AdMob](https://support.google.com/admob/answer/2773509).
   This step creates an AdMob app with a unique [AdMob App
   ID](https://support.google.com/admob/answer/7356431) that is needed later in this
   guide.

## Configure your app

1. In your Gradle settings file, include the
   [Google's Maven repository](https://maven.google.com/web/index.html) and
   [Maven central repository](https://search.maven.org/artifact):

   ### Kotlin

   ```kotlin
   pluginManagement {
     repositories {
       google()
       mavenCentral()
       gradlePluginPortal()
     }
   }

   dependencyResolutionManagement {
     repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
     repositories {
       google()
       mavenCentral()
     }
   }

   rootProject.name = "My Application"
   include(":app")
   ```

   ### Groovy

   ```groovy
   pluginManagement {
     repositories {
       google()
       mavenCentral()
       gradlePluginPortal()
     }
   }

   dependencyResolutionManagement {
     repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
     repositories {
       google()
       mavenCentral()
     }
   }

   rootProject.name = "My Application"
   include ':app'
   ```
2. Add the dependencies for GMA Next-Gen SDK to your app-level build file:

   ### Kotlin

   ```kotlin
   dependencies {
     implementation("com.google.android.libraries.ads.mobile.sdk:ads-mobile-sdk:1.7.0")
   }
   ```

   ### Groovy

   ```groovy
   dependencies {
     implementation 'com.google.android.libraries.ads.mobile.sdk:ads-mobile-sdk:1.7.0'
   }
   ```
3. Click **Sync Now** . For details on syncing, see
   [Sync projects with Gradle files](https://developers.android.com/build#sync-files).

## Initialize the GMA Next-Gen SDK

> [!IMPORTANT]
> **Key Point** : You must initialize GMA Next-Gen SDK before loading ads and interacting with other [`MobileAds`](https://developers.google.com/admob/android/next-gen/reference/kotlin/com/google/android/libraries/ads/mobile/sdk/MobileAds.md.txt) methods, unless explicitly noted in API reference docs (see [`getVersion()`](https://developers.google.com/admob/android/next-gen/reference/kotlin/com/google/android/libraries/ads/mobile/sdk/MobileAds.md.txt#getVersion())). Otherwise, an [`UninitializedPropertyAccessException`](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin/-uninitialized-property-access-exception/) may be thrown.

Call
[`MobileAds.initialize()`](https://developers.google.com/admob/android/next-gen/reference/kotlin/com/google/android/libraries/ads/mobile/sdk/MobileAds.md.txt#initialize(android.content.Context,com.google.android.libraries.ads.mobile.sdk.initialization.InitializationConfig,com.google.android.libraries.ads.mobile.sdk.initialization.OnInitializationCompleteListener))
to initialize GMA Next-Gen SDK. This must be called on a background thread, failure to do
so may cause an "Application Not Responding" (ANR) error.

### Kotlin

```kotlin
import com.google.android.libraries.ads.mobile.sdk.MobileAds
import com.google.android.libraries.ads.mobile.sdk.initialization.InitializationConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    val backgroundScope = CoroutineScope(Dispatchers.IO)
    backgroundScope.launch {
      // Initialize GMA Next-Gen SDK on a background thread.
      MobileAds.initialize(
        this@MainActivity,
        // Sample AdMob app ID: ca-app-pub-3940256099942544~3347511713
        InitializationConfig.Builder("SAMPLE_APP_ID").build()
      ) {
        // Adapter initialization is complete.
      }
      // SDK initialization is complete. If you don't want to wait for bidding adapters to finish
      // initializing, start loading ads now.
    }
  }
}
```

### Java

```java
import com.google.android.libraries.ads.mobile.sdk.MobileAds;
import com.google.android.libraries.ads.mobile.sdk.initialization.InitializationConfig;

public class MainActivity extends AppCompatActivity {
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    new Thread(
            () -> {
              // Initialize GMA Next-Gen SDK on a background thread.
              MobileAds.initialize(
                  this,
                  // Sample AdMob app ID: ca-app-pub-3940256099942544~3347511713
                  new InitializationConfig.Builder("SAMPLE_APP_ID")
                      .build(),
                  initializationStatus -> {
                    // Adapter initialization is complete.
                  });
              // SDK initialization is complete. If you don't want to wait for bidding adapters to
              // finish initializing, start loading ads now.
            })
        .start();
  }
}
```

This method initializes the SDK and calls a completion listener once both
GMA Next-Gen SDK and adapter initializations have completed, or after a
30-second timeout. This needs to be done only once, ideally at app launch.

If you're using AdMob
Mediation, wait until the
completion handler is called before loading ads. This ensures that all
mediation adapters are initialized.

> [!NOTE]
> **Note:** Google User Messaging Platform (UMP) SDK requires the app ID in your app's `AndroidManifest.xml` file. For details, see [Add the application ID](https://developers.google.com/admob/android/next-gen/privacy.md.txt#add_the_application_id).

**Ads may be preloaded by GMA Next-Gen SDK or mediation partner SDKs
upon initialization.** If you need to obtain consent from users in the European
Economic Area (EEA), the United Kingdom (UK), or Switzerland, set any
request-specific flags, such as

[`RequestConfiguration.TagForChildDirectedTreatment`](https://developers.google.com/admob/android/next-gen/reference/kotlin/com/google/android/gms/ads/RequestConfiguration.TagForChildDirectedTreatment.md.txt)
or
[`RequestConfiguration.TagForUnderAgeOfConsent`](https://developers.google.com/admob/android/next-gen/reference/kotlin/com/google/android/gms/ads/RequestConfiguration.TagForUnderAgeOfConsent.md.txt),

or
otherwise take action before loading ads, ensure you do so before initializing
GMA Next-Gen SDK.

<br />

## Select an ad format

GMA Next-Gen SDK is now imported and you're ready to implement an ad.
AdMob offers a number of different ad formats, so
you can choose the one that best fits your app's user experience.

### Banner

![](https://developers.google.com/static/admob/images/format-banner.svg)

Banner ad units display rectangular ads that occupy a portion of an app's
layout. They can refresh automatically after a set period of time. This means
users view a new ad at regular intervals, even if they stay on the same
screen in your app. They're also the simplest ad format to implement.

[Implement banner ads](https://developers.google.com/admob/android/next-gen/banner.md.txt)

### Interstitial

![](https://developers.google.com/static/admob/images/format-interstitial.svg)

Interstitial ad units show full-page ads in your app. Place them at natural
breaks and transitions in your app's interface, such as after level completion
in a gaming app.

[Implement interstitial ads](https://developers.google.com/admob/android/next-gen/interstitial.md.txt)

### Native

![](https://developers.google.com/static/admob/images/format-native.svg)

Native ads are ads where you can customize the way assets such as headlines and
calls to action are presented in your apps. By styling the ad yourself, you can
create a natural, unobtrusive ad presentations that can add to a rich user
experience.

[Implement native ads](https://developers.google.com/admob/android/next-gen/native.md.txt)

### Rewarded

![](https://developers.google.com/static/admob/images/format-rewarded.svg)

Rewarded ad units enable users to play games, take surveys, or watch videos to
earn in-app rewards, such as coins, extra lives, or points. You can set
different rewards for different ad units, and specify the reward values and
items the user received.

[Implement rewarded ads](https://developers.google.com/admob/android/next-gen/rewarded.md.txt)

### Rewarded interstitial

![](https://developers.google.com/static/admob/images/format-rewarded-interstitial.svg)

Rewarded interstitial is a new type of incentivized ad format that lets you
offer rewards, such as coins or extra lives, for ads that appear automatically
during natural app transitions.

Unlike rewarded ads, users aren't required to opt in to view a rewarded
interstitial.

Instead of the opt-in prompt in rewarded ads, rewarded interstitials require an
intro screen that announces the reward and gives users a chance to opt out if
they want to do so.

[Implement rewarded interstitial ads](https://developers.google.com/admob/android/next-gen/rewarded-interstitial.md.txt)

### App open

![](https://developers.google.com/static/admob/images/format-app-open.svg)

App open is an ad format that appears when users open or switch back to your
app. The ad overlays the loading screen.

[Implement app open ads](https://developers.google.com/admob/android/next-gen/app-open.md.txt)

### Picture-in-picture (beta)

![Picture-in-picture](https://developers.google.com/static/admob/images/format-picture-in-picture.svg)

Picture-in-picture (PiP) ads display an ad in a floating window that stays on
top of content on the screen. This format lets your users interact with your app
while the ad remains visible.

[Implement picture-in-picture ads](https://developers.google.com/admob/android/next-gen/picture-in-picture.md.txt)

### Squeezeback (beta)

![Squeezeback](https://developers.google.com/static/admob/images/format-squeezeback.svg)

Squeezeback ads inset your app content into a smaller view and display an
ad in the surrounding space. This format lets your users interact with your app
while the ad remains visible.

[Implement squeezeback ads](https://developers.google.com/admob/android/next-gen/squeezeback.md.txt)
