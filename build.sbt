import sbt._
import sbt.Keys._

val cogcompNLPVersion = "3.1.25"
val ccgGroupId = "edu.illinois.cs.cogcomp"

lazy val commonSettings = Seq(
  organization := ccgGroupId,
  version := "1.5",
  scalaVersion := "2.11.8",
  javaOptions ++= Seq("-Xmx10G"),
  fork := false,
  // Make sure SCIP libraries are locatable.
  javaOptions += s"-Djava.library.path=lib",
  envVars ++= Map(
    "LD_LIBRARY_PATH" -> "lib",
    "DYLD_LIBRARY_PATH" -> "lib"
  ),
  includeFilter in unmanagedJars := "*.jar" || "*.so" || "*.dylib"
)

// TODO(danm): This is used enough projects to be in allenai/sbt-plugins CoreDependencies.
def nlpstack(component: String) = ("org.allenai.nlpstack" %% s"nlpstack-$component" % "1.6") // exclude("org.slf4j", "log4j-over-slf4j")
  .exclude("commons-logging", "commons-logging")
  .exclude("edu.stanford.nlp", "stanford-corenlp")
  .exclude("org.slf4j", "log4j-over-slf4j")

def textualEntailment(component: String) = ("org.allenai.textual-entailment" %% component % "1.0.6")
  .exclude("org.slf4j", "log4j-over-slf4j")
  .exclude("edu.stanford.nlp", "stanford-corenlp")

def ccgLib(component: String) = (ccgGroupId % component % cogcompNLPVersion withSources).exclude("edu.cmu.cs.ark", "ChuLiuEdmonds")

val sprayVersion = "1.3.3"
def sprayModule(id: String): ModuleID = "io.spray" %% s"spray-$id" % sprayVersion
val sprayClient = sprayModule("client")

lazy val envUser = System.getenv("COGCOMP_USER")
lazy val user = if (envUser == null) System.getProperty("user.name") else envUser
lazy val keyFile = new java.io.File(Path.userHome.absolutePath + "/.ssh/id_rsa")

lazy val publishSettings = Seq(
  publishTo := Some(
    Resolver.ssh(
      "CogcompSoftwareRepo", "bilbo.cs.illinois.edu",
      "/mounts/bilbo/disks/0/www/cogcomp/html/m2repo/") as(user, keyFile)
  )
)

lazy val root = (project in file(".")).
  //enablePlugins(StylePlugin).
  settings(commonSettings: _*).
  settings(publishSettings: _*).
  settings(
    name := "text-ilp",
    libraryDependencies ++= Seq(
      textualEntailment("interface"),
      textualEntailment("service"),
      "io.spray" % "spray-caching_2.11" % "1.3.3",
      "org.allenai.common" %% "common-core" % "1.4.6",
      "org.allenai.common" %% "common-cache" % "1.4.6",
      "commons-io" % "commons-io" % "2.4",
      "net.sf.opencsv" % "opencsv" % "2.1",
      "com.typesafe.play" % "play-json_2.11" % "2.5.9",
      "org.rogach" %% "scallop" % "2.0.5",
      "com.google.inject" % "guice" % "4.0",
      "net.debasishg" %% "redisclient" % "3.0",
      "com.medallia.word2vec" % "Word2VecJava" % "0.10.3",
      ccgLib("illinois-core-utilities"),
      ccgLib("illinois-inference"),
      ccgLib("illinois-nlp-pipeline"),
      ccgLib("illinois-curator"),
      "edu.cmu.cs.ark" % "ChuLiuEdmonds" % "1.0" force(),
      ccgGroupId % "scip-jni" % "3.1.1",
      "edu.cmu" % "Meteor" % "1.5",
      "org.slf4j" % "slf4j-log4j12" % "1.7.12",
      nlpstack("chunk"),
      nlpstack("lemmatize"),
      nlpstack("tokenize"),
      nlpstack("postag"),
      nlpstack("core"),
      nlpstack("parse"),
      sprayClient,
      "org.scalatest" % "scalatest_2.11" % "2.2.4",
      "org.elasticsearch" % "elasticsearch" % "2.4.1",
      "me.tongfei" % "progressbar" % "0.5.1",
      "org.scalaz" %% "scalaz-core" % "7.2.8",
      "com.github.mpkorstanje" % "simmetrics-core" % "4.1.1",
      "github.sahand" % "sahand-client_2.11" % "1.2.2",
      "io.github.pityka" %% "nspl-awt" % "0.0.7",
      "nz.ac.waikato.cms.weka" % "weka-dev" % "3.7.12",
      "edu.mit" % "jverbnet" % "1.2.0.1",
      "commons-codec" % "commons-codec" % "1.11"
      //"com.twitter" % "chill_2.11" % "0.5.1"
    ),
    resolvers ++= Seq(
      Resolver.mavenLocal,
      Resolver.bintrayRepo("allenai", "maven"),
      Resolver.bintrayRepo("allenai", "private"),
      "CogcompSoftware" at "https://cogcomp.seas.upenn.edu/m2repo/"
    )
  )

lazy val viz = (project in file("viz")).
  settings(commonSettings: _*).
  dependsOn(root).
  aggregate(root).
  enablePlugins(PlayScala).
  disablePlugins(PlayLogback).
  settings(
    name := "text-ilp-visualization",
    libraryDependencies ++= Seq(
      filters,
      "org.scalatestplus.play" %% "scalatestplus-play" % "1.5.1" % Test,
      "com.typesafe.play" % "play_2.11" % "2.5.10",
      "org.webjars" %% "webjars-play" % "2.4.0-1",
      "org.webjars" % "bootstrap" % "3.3.7",
      "org.webjars" % "jquery" % "3.1.1",
      "org.webjars" % "headjs" % "1.0.3"
    ),
    resolvers ++= Seq("scalaz-bintray" at "http://dl.bintray.com/scalaz/releases") //,
  )
