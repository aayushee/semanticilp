package org.allenai.ari.solvers.textilp

import edu.illinois.cs.cogcomp.core.datastructures.ViewNames
import org.allenai.ari.solvers.textilp.alignment.AlignmentFunction
import org.allenai.ari.solvers.textilp.solvers.{LuceneSolver, SalienceSolver, TextILPSolver, TextSolver}
import org.allenai.ari.solvers.textilp.utils.{AnnotationUtils, Constants, SQuADReader, SolverUtils}
import org.rogach.scallop._

object ExperimentsApp {
  lazy val annotationUtils = new AnnotationUtils()
  lazy val textILPSolver = new TextILPSolver(annotationUtils)
  lazy val salienceSolver = new SalienceSolver()
  lazy val luceneSolver = new LuceneSolver()

  class ArgumentParser(args: Array[String]) extends ScallopConf(args) {
    val experimentType: ScallopOption[Int] = opt[Int]("type", descr = "Experiment type", required = true)
    verify()
  }

  def testQuantifier(): Unit = {
    val ta = annotationUtils.pipelineService.createAnnotatedTextAnnotation("", "",
      "The annual NFL Experience was held at the Moscone Center in San Francisco. In addition, \"Super Bowl City\" opened on January 30 at Justin Herman Plaza on The Embarcadero, featuring games and activities that will highlight the Bay Area's technology, culinary creations, and cultural diversity. More than 1 million people are expected to attend the festivities in San Francisco during Super Bowl Week. San Francisco mayor Ed Lee said of the highly visible homeless presence in this area \"they are going to have to leave\". San Francisco city supervisor Jane Kim unsuccessfully lobbied for the NFL to reimburse San Francisco for city services in the amount of $5 million.")
    annotationUtils.quantifierAnnotator.addView(ta)
    println(ta)
    println(ta.getAvailableViews)
  }

  def testPipelineAnnotation(): Unit = {
    val ta = annotationUtils.pipelineService.createAnnotatedTextAnnotation("", "",
      "this is a sample senrence that needs to be update with 20 pipelines in Illinois. ")
    println(ta)
    println(ta.getAvailableViews)
  }

  def dumpSQuADQuestionsOnDisk(reader: SQuADReader) = {
    import java.io._
    val pw = new PrintWriter(new File("squadQuestions.txt" ))
    reader.instances.zipWithIndex.foreach {
      case (ins, idx) =>
        ins.paragraphs.foreach { p =>
          pw.write("P: " + p.context + "\n")
          p.questions.foreach { q =>
            pw.write("Q: " + q.questionText + "\n")
          }
        }
    }
    pw.close()
  }

  def testCuratorAnnotation() = {
    val ta = annotationUtils.curatorService.createBasicTextAnnotation("", "", "This is a sample sentence. Barak Obama is the president of US. Lake Tahoe is a nice place to visit. I like the blue skye. ")
    annotationUtils.curatorService.addView(ta, ViewNames.WIKIFIER)
//    annotationUtils.curatorService.addView(ta, ViewNames.WIKIFIER)
    println(ta.getAvailableViews)
  }

  def processSQuADWithWikifier(reader: SQuADReader) = {
    import java.io._
    val pw = new PrintWriter(new File("squadTrain.txt" ))
    reader.instances.zipWithIndex.foreach {
      case (ins, idx) =>
        ins.paragraphs.foreach { p =>
          pw.write("P: " + p.context + "\n")
          p.questions.foreach { q =>
            pw.write("Q: " + q.questionText + "\n")
          }
        }
    }
    pw.close()
  }

  //TODO if "the" is among the candidate answrs, drop it and make it another candidate
  //TODO capture aphabetical numbers too, like "six"
  def generateCandiateAnswers(reader: SQuADReader): Unit = {
    var found: Int = 0
    var notFound: Int = 0
    reader.instances.zipWithIndex.foreach {
      case (ins, idx) =>
        println("Idx: " + idx + " / ratio: " + idx * 1.0 / reader.instances.size)
        ins.paragraphs.foreach { p =>
          p.contextTAOpt match {
            case None => throw new Exception("The instance does not contain annotation . . . ")
            case Some(annotation) =>
              val candidateAnswers = SolverUtils.getCandidateAnswer(annotation)
              p.questions.foreach { q =>
                val goldAnswers = q.answers.map(_.answerText)
                if (goldAnswers.exists(candidateAnswers.contains)) {
                  println(" --> found ")
                  found = found + 1
                } else {
                  notFound = notFound + 1
                  println(" --> not found ")
                  println("Question: " + q)
                  println("CandidateAnswers: " + candidateAnswers)
                  println("context = " + p.context)
                }
              }
          }
        }
    }
    println("found: " + found + "\nnot-found: " + notFound)
  }

  def evaluateDataSetWithRemoteSolver(reader: SQuADReader, solver: String): Unit = {
    reader.instances.slice(0, 3).zipWithIndex.foreach {
      case (ins, idx) =>
        println("Idx: " + idx + " / ratio: " + idx * 1.0 / reader.instances.size)
        ins.paragraphs.slice(0, 3).foreach { p =>
          p.contextTAOpt match {
            case None => throw new Exception("The instance does not contain annotation . . . ")
            case Some(annotation) =>
              val candidateAnswers = SolverUtils.getCandidateAnswer(annotation).toSeq
              p.questions.foreach { q =>
                val goldAnswers = q.answers.map(_.answerText)
                val perOptionScores = SolverUtils.handleQuestionWithManyCandidates(q.questionText, candidateAnswers, solver)
                println("q.questionText = " + q.questionText)
                println("gold = " + goldAnswers)
                //println("predicted = " + perOptionScores.sortBy(-_._2))
                println("---------")
              }
          }
        }
    }
  }

  def testAlignmentScore() = {
    println("Testing the alignment . . . ")
    println("ent(moon, moon) = " + AlignmentFunction.entailment.entail(Seq("moon"), Seq("moon")))
    println("ent(moon, sun) = " + AlignmentFunction.entailment.entail(Seq("moon"), Seq("sun")))
    println("ent(sun, moon) = " + AlignmentFunction.entailment.entail(Seq("sun"), Seq("moon")))
  }

  def solveSampleQuestionWithTextILP() = {
    textILPSolver.solve(
      "A decomposer is an organism that",
      Seq("hunts and eats animals", "migrates for the winter",
        "breaks down dead plants and animals", "uses water and sunlight to make food"),
      "explanation:Decomposers: organisms that obtain energy by eating dead plant or animal matter. " +
        "Windy, cloudy, rainy, and cold are words that help describe\tfocus: deposition. " +
        "explanation:DECOMPOSER An organism that breaks down cells of dead plants and animals into simpler substances." +
        "explanation:The plants use sunlight, carbon dioxide, water, and minerals to make food that sustains themselves and other organisms in the forest."
    )
//    textILPSolver.solve(
//      "A decomposer",
//      Set("hunts ", "migrates for the winter",
//        "breaks down dead plants and animals", "uses water and sunlight to make food"),
//      "Decomposers"
//    )
  }

  def testElasticSearchSnippetExtraction() = {
    println(SolverUtils.extractParagraphGivenQuestionAndFocusWord("when is the best time of the year in New York city, especially when it snows or rains?", "Christmas", 3).mkString("\n"))
  }

  def extractKnowledgeSnippet() = {
    val question = "In New York State the longest period of daylight occurs during which month"
    val candidates = Seq("June", "March", "December", "September")
    candidates.foreach { focus =>
      println("Query: " + question + "  " + focus + " --> Result: " + SolverUtils.extractParagraphGivenQuestionAndFocusWord2(question, focus, 3).mkString(" "))
      //println("Q: In New York State --> " + SolverUtils.extractParagraphGivenQuestionAndFocusWord2("In New York State ", focus, 3))
      //println("Q: the longest period of daylight --> " +SolverUtils.extractParagraphGivenQuestionAndFocusWord2("the longest period of daylight", focus, 3))
      //println("----------")
    }
    println("Query: " + question + "  " + candidates + " --> Result: " + SolverUtils.extractPatagraphGivenQuestionAndFocusSet3(question, candidates, 8).mkString(" "))
  }

  def testRemoteSolverWithSampleQuestion() = {
    SolverUtils.evaluateASingleQuestion("Which two observations are both used to describe weather? (A) like (B) the difference (C) events (D) temperature and sky condition", "tableilp")
  }

  def evaluateTextSolverOnRegents(dataset: Seq[(String, Seq[String], String)], textSolver: TextSolver) = {
    SolverUtils.printMemoryDetails()
    println("Starting the evaluation . . . ")
    val perQuestionScore = dataset.map{ case (question, options, correct) =>
      //println("collecting knolwdge . . . ")
//      val knowledgeSnippet = options.flatMap(focus => SolverUtils.extractParagraphGivenQuestionAndFocusWord(question, focus, 3)).mkString(" ")
//      val knowledgeSnippet = options.flatMap(focus => SolverUtils.extractParagraphGivenQuestionAndFocusWord2(question, focus, 3)).mkString(" ")
      val knowledgeSnippet = if(textSolver.isInstanceOf[TextILPSolver]) {
        SolverUtils.extractPatagraphGivenQuestionAndFocusSet3(question, options, 8).mkString(" ")
      }
      else {
        ""
      }
      //println("solving it . . . ")
      val (selected, _) = textSolver.solve(question, options, knowledgeSnippet)
      val score = SolverUtils.assignCredit(selected, correct.head - 'A', options.length)
      //println("Question: " + question + " / options: " + options  +  "   / selected: " + selected  + " / score: " + score)
      score
    }
    println("Average score: " + perQuestionScore.sum / perQuestionScore.size)
  }

  def evaluateTextilpOnSquad(reader: SQuADReader) = {
    SolverUtils.printMemoryDetails()
    val (exactMatch, f1, total) = reader.instances.slice(0, 30).zipWithIndex.flatMap {
      case (ins, idx) =>
        println("Idx: " + idx + " / ratio: " + idx * 1.0 / reader.instances.size)
        ins.paragraphs.slice(0, 5).flatMap { p =>
          p.contextTAOpt match {
            case None => throw new Exception("The instance does not contain annotation . . . ")
            case Some(annotation) =>
              val candidateAnswers = SolverUtils.getCandidateAnswer(annotation).toSeq
              p.questions.slice(0, 1).map { q =>
                val (selected, _) = textILPSolver.solve(q.questionText, candidateAnswers, p.context)
                SolverUtils.assignCreditSquad(candidateAnswers(selected.head), q.answers.map(_.answerText))
              }
          }
        }
    }.unzip3
    println("Average exact match: " + exactMatch.sum / total.sum + "  /   Average f1: " + f1.sum / total.sum)
  }

  def testTheDatastes() = {
    println("omnibusTrain: " + SolverUtils.omnibusTrain.length)
    println("omnibusTest: " + SolverUtils.omnibusTest.length)
    println("omnibusDev: " + SolverUtils.omnibusDev.length)
    println("publicTrain: " + SolverUtils.publicTrain.length)
    println("publicTest: " + SolverUtils.publicTest.length)
    println("publicDev: " + SolverUtils.publicDev.length)
    println("regentsTrain: " + SolverUtils.regentsTrain.length)
  }

  def testSquadPythonEvaluationScript() = {
    println(SolverUtils.assignCreditSquad("the country in the east", Seq("east", "world")))
  }

  def main(args: Array[String]): Unit = {
    lazy val trainReader = new SQuADReader(Constants.squadTrainingDataFile, Some(annotationUtils.pipelineService), annotationUtils)
    lazy val devReader = new SQuADReader(Constants.squadDevDataFile, Some(annotationUtils.pipelineService), annotationUtils)
    val parser = new ArgumentParser(args)
    parser.experimentType() match {
      case 1 => generateCandiateAnswers(devReader)
      case 2 => testQuantifier()
      case 3 => testPipelineAnnotation()
      case 4 => testRemoteSolverWithSampleQuestion()
      case 5 => evaluateDataSetWithRemoteSolver(devReader, "salience")
      case 6 => solveSampleQuestionWithTextILP()
      case 7 => testAlignmentScore()
      case 8 => testElasticSearchSnippetExtraction()
      case 9 => testTheDatastes()
      case 10 =>
        evaluateTextSolverOnRegents(SolverUtils.publicTrain, luceneSolver)
        evaluateTextSolverOnRegents(SolverUtils.publicTest, luceneSolver)
        evaluateTextSolverOnRegents(SolverUtils.publicTrain, salienceSolver)
        evaluateTextSolverOnRegents(SolverUtils.publicTest, salienceSolver)

      case 11 =>
//        evaluateTextilpOnRegents(SolverUtils.publicTrain)
//        println("==== public train ")
//        evaluateTextilpOnRegents(SolverUtils.publicDev)
//        println("==== public dev ")
//        evaluateTextilpOnRegents(SolverUtils.publicTest)
//        println("==== public test ")
        evaluateTextSolverOnRegents(SolverUtils.regentsTrain, textILPSolver)
//        println("==== regents train  ")
      case 12 => extractKnowledgeSnippet()
      case 13 => testSquadPythonEvaluationScript()
      case 14 => evaluateTextilpOnSquad(trainReader)
      case 15 => dumpSQuADQuestionsOnDisk(devReader)
      case 16 => testCuratorAnnotation()
      case 17 => processSQuADWithWikifier(devReader)
    }
  }
}