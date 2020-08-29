package org.allenai.ari.solvers.textilp.solvers


package org.allenai.ari.solvers.textilp.solvers

import java.io.File
import java.net.URL

import edu.cmu.meteor.scorer.{MeteorConfiguration, MeteorScorer}
import edu.illinois.cs.cogcomp.core.datastructures.ViewNames
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.{Constituent, PredicateArgumentView, TextAnnotation}
import github.sahand.{SahandClient, SimilarityNames}
import org.allenai.ari.controller.questionparser.{FillInTheBlankGenerator, QuestionParse}
import org.simmetrics.StringMetric
import org.simmetrics.metrics.StringMetrics

import scala.collection.mutable
import org.allenai.ari.solvers.bioProccess.ProcessBankReader._
import org.allenai.ari.solvers.textilp.{EntityRelationResult, _}
import org.allenai.ari.solvers.textilp.alignment.{AlignmentFunction, KeywordTokenizer}
import org.allenai.ari.solvers.textilp.ilpsolver.{IlpVar, _}
import org.allenai.ari.solvers.textilp.utils.{AnnotationUtils, Constants, SolverUtils}
import weka.classifiers.Classifier
import weka.core.converters.ArffLoader
import weka.core.{DenseInstance, Instance, Instances, SparseInstance}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

trait ReasoningType {}

case object SimpleMatching extends ReasoningType

trait TextILPModel {}

object TextILPModel {
  case object MyModel extends TextILPModel

}

object SingleSolver {
  val pathLSTMViewName = "SRL_VERB_PATH_LSTM"
  val stanfordCorefViewName = "STANFORD_COREF"
  val curatorSRLViewName = "SRL_VERB_CURATOR"
  val clausIeViewName = "CLAUSIE"

  val epsilon = 0.001
  val oneActiveSentenceConstraint = true

  val scienceTermsBoost = false
  val interSentenceAlignments = false
  val stopWords = false
  val essentialTerms = false
  val minInterSentenceAlignmentScore = 0.0
  val activeSentenceAlignmentCoeff = -1.0
  // penalizes extra sentence usage
  val constituentAlignmentCoeff = -0.1
  val activeScienceTermBoost = 1d
  val minActiveParagraphConstituentAggrAlignment = 0.1
  val minActiveQuestionConstituentAggrAlignment = 0.1
  val minAlignmentWhichTerm = 0.6d
  val minPConsToPConsAlignment = 0.6
  val minPConsTQChoiceAlignment = 0.2
  val whichTermAddBoost = 1.5d
  val whichTermMulBoost = 1d

  val essentialTermsMturkConfidenceThreshold = 0.9
  val essentialClassifierConfidenceThreshold = 0.9
  val essentialTermsFracToCover = 1.0
  // a number in [0,1]
  val essentialTermsSlack = 1
  // a non-negative integer
  val essentialTermWeightScale = 1.0
  val essentialTermWeightBias = 0.0
  val essentialTermMinimalSetThreshold = 0.8
  val essentialTermMaximalSetThreshold = 0.2
  val essentialTermMinimalSetTopK = 3
  val essentialTermMaximalSetBottomK = 0
  val essentialTermMinimalSetSlack = 1
  val essentialTermMaximalSetSlack = 0
  val trueFalseThreshold = 5.5 // this has to be tuned

  lazy val keywordTokenizer = KeywordTokenizer.Default

  // fill-in-blank generator
  lazy val fitbGenerator = FillInTheBlankGenerator.mostRecent


  lazy val offlineAligner = new AlignmentFunction("Entailment", 0.2,
    TextILPSolver.keywordTokenizer, useRedisCache = false, useContextInRedisCache = false)

  lazy val sahandClient = new SahandClient(s"${Constants.sahandServer}:${Constants.sahandPort}/")

  val toBeVerbs = Set("am", "is", "are", "was", "were", "being", "been", "be", "were", "be")
}

case class TextIlpParams(
                          activeQuestionTermWeight: Double,
                          alignmentScoreDiscount: Double,
                          questionCellOffset: Double,
                          paragraphAnswerOffset: Double,
                          firstOrderDependencyEdgeAlignments: Double,
                          activeParagraphConstituentsWeight: Double,

                          minQuestionTermsAligned: Int,
                          maxQuestionTermsAligned: Int,
                          minQuestionTermsAlignedRatio: Double,
                          maxQuestionTermsAlignedRatio: Double,

                          activeSentencesDiscount: Double,
                          maxActiveSentences: Int,

                          longerThan1TokenAnsPenalty: Double,
                          longerThan2TokenAnsPenalty: Double,
                          longerThan3TokenAnsPenalty: Double,

                          exactMatchMinScoreValue: Double,
                          exactMatchMinScoreDiff: Double,
                          exactMatchSoftWeight: Double, // supposed to be a positive value

                          meteorExactMatchMinScoreValue: Double,
                          meteorExactMatchMinScoreDiff: Double,

                          minQuestionToParagraphAlignmentScore: Double,
                          minParagraphToQuestionAlignmentScore: Double,

                          // Answers: sparsity
                          moreThan1AlignmentAnsPenalty: Double,
                          moreThan2AlignmentAnsPenalty: Double,
                          moreThan3AlignmentAnsPenalty: Double,

                          // Question: sparsity
                          moreThan1AlignmentToQuestionTermPenalty: Double,
                          moreThan2AlignmentToQuestionTermPenalty: Double,
                          moreThan3AlignmentToQuestionTermPenalty: Double,

                          // Paragraph: proximity inducing
                          activeDist1WordsAlignmentBoost: Double,
                          activeDist2WordsAlignmentBoost: Double,
                          activeDist3WordsAlignmentBoost: Double,

                          // Paragraph: sparsity
                          maxNumberOfWordsAlignedPerSentence: Int,
                          maxAlignmentToRepeatedWordsInParagraph: Int,
                          moreThan1AlignmentToParagraphTokenPenalty: Double,
                          moreThan2AlignmentToParagraphTokenPenalty: Double,
                          moreThan3AlignmentToParagraphTokenPenalty: Double,

                          // Paragraph: intra-sentence alignment
                          coreferenceWeight: Double,
                          intraSentenceAlignmentScoreDiscount: Double,
                          entailmentWeight: Double,
                          srlAlignmentWeight: Double,
                          scieneTermBoost: Double
                        )

class SingleSolver(annotationUtils: AnnotationUtils,
                    verbose: Boolean = false, params: TextIlpParams,
                    useRemoteAnnotation: Boolean = true) extends TextSolver {

  lazy val aligner = new AlignmentFunction("Entailment", 0.0,
    TextILPSolver.keywordTokenizer, useRedisCache = false, useContextInRedisCache = false)


  def solve(question: String, options: Seq[String], snippet: String): (Seq[Int], EntityRelationResult) = {
    val (q: Question, p: Paragraph) = preprocessQuestionData(question, options, snippet)
    require(p.contextTAOpt.isDefined, "pTA is not defined after pre-processing . . . ")
    require(q.qTAOpt.isDefined, "qTA is not defined after pre-processing . . . ")
    println("Reasoning methods . . . ")


    lazy val resultSimpleMatching = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(SimpleMatching), useSummary = true, TextILPSolver.pathLSTMViewName)
      ilpSolver.free()
      result
    } -> "resultILP"



    val resultOpt = Constants.textILPModel match {
      case TextILPModel.MyModel =>
        (resultSimpleMatching #:: Stream.empty).find { t =>
          println("trying: " + t._2)
          t._1._1.nonEmpty
        }
    }

    if (resultOpt.isDefined) {
      println(" ----> Selected method: " + resultOpt.get._2)
      resultOpt.get._1
    }
    else (Seq.empty, EntityRelationResult())
  }



  def solveWithHighestScore(question: String, options: Seq[String], snippet: String): (Seq[Int], EntityRelationResult) = {
    val (q: Question, p: Paragraph) = preprocessQuestionData(question, options, snippet)
    val types = Seq(SimpleMatching, VerbSRLandPrepSRL, SRLV1ILP, VerbSRLandCoref /*, VerbSRLandCommaSRL*/)
    val srlViewsAll = Seq(ViewNames.SRL_VERB /*, TextILPSolver.curatorSRLViewName, TextILPSolver.pathLSTMViewName*/)
    //val srlViewsAll = Seq(SimpleMatching)
    val scores = types.flatMap { t =>
      val start = System.currentTimeMillis()
      SolverUtils.printMemoryDetails()
      val srlViewTypes = if (t == CauseRule || t == WhatDoesItDoRule || t == SimpleMatching) Seq(TextILPSolver.pathLSTMViewName) else srlViewsAll
      srlViewTypes.map { srlViewName =>
        t match {
          case SRLV1Rule => SRLSolverV1(q, p, srlViewName)
          case SRLV2Rule => SRLSolverV2(q, p, srlViewName)
          case SRLV3Rule => SRLSolverV3(q, p, aligner, srlViewName)
          case WhatDoesItDoRule => WhatDoesItDoSolver(q, p)
          case CauseRule => CauseResultRules(q, p)
          case x =>
            val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
            val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(x), useSummary = true, srlViewName)
            ilpSolver.free()
            result
        }
      }
    }
    if (scores.exists(_._1.nonEmpty)) {
      scores.maxBy(_._2.statistics.objectiveValue)
    }
    else {
      (Seq.empty, EntityRelationResult())
    }
  }

  def preprocessQuestionData(question: String, options: Seq[String], snippet1: String): (Question, Paragraph) = {
    val snippet = {
      val cleanQ = SolverUtils.clearRedundantCharacters(question)
      val questionTA = try {
        annotationUtils.pipelineServerClientWithBasicViews.annotate(cleanQ)
      }
      catch {
        case e: Exception =>
          println((s"\nOriginal questions: ${question} \ncleanQ: ${cleanQ}"))
          e.printStackTrace()
          throw new Exception
      }

      val cleanSnippet = SolverUtils.clearRedundantCharacters(snippet1)
      val paragraphTA = try {
        annotationUtils.pipelineServerClientWithBasicViews.annotate(cleanSnippet)
      }
      catch {
        case e: Exception =>
          println("pipelineServerClientWithBasicViews failed on the following input: ")
          println((s"\nsnippet1: ${snippet1} \ncleanSnippet: ${cleanSnippet}"))
          e.printStackTrace()
          throw new Exception
      }
      SolverUtils.ParagraphSummarization.getSubparagraphString(paragraphTA, questionTA)
    }
    println("pre-processsing . . .  ")
    val answers = options.map { o =>
      val ansTA = try {
        val ta = annotationUtils.pipelineServerClient.annotate(o)
        Some(ta)
      } catch {
        case e: Exception =>
          e.printStackTrace()
          None
      }
      Answer(o, -1, ansTA)
    }
    println("Annotating question: ")

    val qTA = if (question.trim.nonEmpty) {
      Some(annotationUtils.annotateWithEverything(question, withFillInBlank = true))
    }
    else {
      println("Question string is empty . . . ")
      println("question: " + question)
      println("snippet: " + snippet)
      None
    }
    val q = Question(question, "", answers, qTA)
    val pTA = if (snippet.trim.nonEmpty) {
      try {
        Some(annotationUtils.annotateWithEverything(snippet, withFillInBlank = false))
      }
      catch {
        case e: Exception =>
          e.printStackTrace()
          None
      }
    }
    else {
      println("Paragraph String is empty ..... ")
      println("question: " + question)
      println("snippet: " + snippet)
      None
    }
    val p = Paragraph(snippet, Seq(q), pTA)
    (q, p)
  }


  def createILPModel[V <: IlpVar](
                                   q: Question,
                                   p: Paragraph,
                                   ilpSolver: IlpSolver[V, _],
                                   alignmentFunction: AlignmentFunction,
                                   reasoningTypes: Set[ReasoningType],
                                   useSummary: Boolean,
                                   srlViewName: String
                                 ): (mutable.Buffer[(Constituent, Constituent, V)], mutable.Buffer[(Constituent, Int, Int, V)], mutable.Buffer[(Constituent, Constituent, V)], Seq[(Int, V)], Boolean, Seq[Seq[String]]) = {
    if (verbose) println("starting to create the model  . . . ")
    val isTrueFalseQuestion = q.isTrueFalse

    val isTemporalQuestions = q.isTemporal
    require(q.qTAOpt.isDefined, "the annotatins for the question is not defined")
    require(p.contextTAOpt.isDefined, "the annotatins for the paragraph is not defined")
    q.answers.foreach { a =>
      require(a.aTAOpt.isDefined, s"the annotations for answers is not defined . . . ")
      require(a.aTAOpt.get.hasView(ViewNames.SHALLOW_PARSE), s"the shallow parse view is not defined; view: " + a.aTAOpt.get.getAvailableViews.asScala)
    }
    val qTA = q.qTAOpt.getOrElse(throw new Exception("The annotation for the question not found . . . "))
    val pTA = p.contextTAOpt.getOrElse(throw new Exception("The annotation for the paragraph not found . . . "))
    val qTokens = if (qTA.hasView(ViewNames.SHALLOW_PARSE)) qTA.getView(ViewNames.SHALLOW_PARSE).getConstituents.asScala else Seq.empty
    val pTokens = if (pTA.hasView(ViewNames.SHALLOW_PARSE)) pTA.getView(ViewNames.SHALLOW_PARSE).getConstituents.asScala else Seq.empty

    def getParagraphConsCovering(c: Constituent): Option[Constituent] = {
      p.contextTAOpt.get.getView(ViewNames.SHALLOW_PARSE).getConstituentsCovering(c).asScala.headOption
    }

    ilpSolver.setAsMaximization()

    var interParagraphAlignments: mutable.Buffer[(Constituent, Constituent, V)] = mutable.Buffer.empty
    var questionParagraphAlignments: mutable.Buffer[(Constituent, Constituent, V)] = mutable.Buffer.empty
    var paragraphAnswerAlignments: mutable.Buffer[(Constituent, Int, Int, V)] = mutable.Buffer.empty

    // whether to create the model with the tokenized version of the answer options
    val tokenizeAnswers = true
    //if(q.isTemporal) true else false
    val aTokens = if (tokenizeAnswers) {
      q.answers.map(_.aTAOpt.get.getView(ViewNames.SHALLOW_PARSE).getConstituents.asScala.map(_.getSurfaceForm))
    }
    else {
      q.answers.map(a => Seq(a.answerText))
    }

    def getAnswerOptionCons(ansIdx: Int, ansTokIdx: Int): Constituent = {
      q.answers(ansIdx).aTAOpt.get.getView(ViewNames.SHALLOW_PARSE).getConstituents.get(ansTokIdx)
    }

    // high-level variables
    // active answer options
    val activeAnswerOptions = if (!isTrueFalseQuestion) {
      for {
        ansIdx <- q.answers.indices
        x = ilpSolver.createBinaryVar("activeAnsOptId" + ansIdx, 0.0)
      } yield (ansIdx, x)
    }
    else {
      List.empty
    }

    def getVariablesConnectedToOptionToken(ansIdx: Int, tokenIdx: Int): Seq[V] = {
      paragraphAnswerAlignments.filter { case (_, ansIdxTmp, tokenIdxTmp, _) =>
        ansIdxTmp == ansIdx && tokenIdxTmp == tokenIdx
      }.map(_._4)
    }

    def getVariablesConnectedToOption(ansIdx: Int): Seq[V] = {
      paragraphAnswerAlignments.filter { case (_, ansTmp, _, _) => ansTmp == ansIdx }.map(_._4)
    }

    println("reasoningTypes: " + reasoningTypes)

    if (reasoningTypes.contains(SimpleMatching)) {
      // create questionToken-paragraphToken alignment edges
      val questionTokenParagraphTokenAlignments = for {
        qCons <- qTokens
        pCons <- pTokens
        // TODO: make it QuestionCell score
        //      score = alignmentFunction.scoreCellCell(qCons.getSurfaceForm, pCons.getSurfaceForm) + params.questionCellOffset
        score = alignmentFunction.scoreCellQCons(pCons.getSurfaceForm, qCons.getSurfaceForm) + params.questionCellOffset
        if score > params.minParagraphToQuestionAlignmentScore
        x = ilpSolver.createBinaryVar("", score)
      } yield (qCons, pCons, x)

      questionParagraphAlignments ++= questionTokenParagraphTokenAlignments.toBuffer

      // create paragraphToken-answerOption alignment edges
      val paragraphTokenAnswerAlignments = if (!isTrueFalseQuestion) {
        // create only multiple nodes at each answer option
        for {
          pCons <- pTokens
          ansIdx <- aTokens.indices
          ansConsIdx <- aTokens(ansIdx).indices
          ansConsString = aTokens(ansIdx).apply(ansConsIdx)
          // TODO: make it QuestionCell score
          score = alignmentFunction.scoreCellCell(pCons.getSurfaceForm, ansConsString) + params.paragraphAnswerOffset
          //        score = alignmentFunction.scoreCellQChoice(pCons.getSurfaceForm, ansConsString) + params.paragraphAnswerOffset
          x = ilpSolver.createBinaryVar("", score)
        } yield (pCons, ansIdx, ansConsIdx, x)
      } else {
        List.empty
      }

      paragraphAnswerAlignments ++= paragraphTokenAnswerAlignments.toBuffer

      def getAnswerOptionVariablesConnectedToParagraph(c: Constituent): Seq[(Int, Int, V)] = {
        paragraphTokenAnswerAlignments.filter { case (cTmp, ansIdxTmp, tokenIdxTmp, _) => cTmp == c }.map(tuple => (tuple._2, tuple._3, tuple._4))
      }

      def getVariablesConnectedToParagraphToken(c: Constituent): Seq[V] = {
        questionTokenParagraphTokenAlignments.filter { case (_, cTmp, _) => cTmp == c }.map(_._3) ++
          paragraphTokenAnswerAlignments.filter { case (cTmp, _, _, _) => cTmp == c }.map(_._4)
      }

      def getVariablesConnectedToParagraphSentence(sentenceId: Int): Seq[V] = {
        pTokens.filter(_.getSentenceId == sentenceId).flatMap(getVariablesConnectedToParagraphToken)
      }

      def getVariablesConnectedToQuestionToken(qCons: Constituent): Seq[V] = {
        questionTokenParagraphTokenAlignments.filter { case (cTmp, _, _) => cTmp == qCons }.map(_._3)
      }

      // active paragraph constituent
      val activeParagraphConstituents = pTokens.map { t =>
        t -> ilpSolver.createBinaryVar("", params.activeParagraphConstituentsWeight)
      }.toMap
      // the paragraph token is active if anything connected to it is active
      activeParagraphConstituents.foreach {
        case (ans, x) =>
          val connectedVariables = getVariablesConnectedToParagraphToken(ans)
          val allVars = connectedVariables :+ x
          val coeffs = Seq.fill(connectedVariables.length)(-1.0) :+ 1.0
          ilpSolver.addConsBasicLinear("activeOptionVar", allVars, coeffs, None, Some(0.0))
          connectedVariables.foreach { connectedVar =>
            val vars = Seq(connectedVar, x)
            val coeffs = Seq(1.0, -1.0)
            ilpSolver.addConsBasicLinear("activeParagraphConsVar", vars, coeffs, None, Some(0.0))
          }
      }

      // active sentences for the paragraph
      val activeSentences = for {
        s <- 0 until pTA.getNumberOfSentences
        // alignment is preferred for lesser sentences; hence: negative activeSentenceDiscount
        x = ilpSolver.createBinaryVar("activeSentence:" + s, params.activeSentencesDiscount)
      } yield (s, x)
      // the paragraph constituent variable is active if anything connected to it is active
      activeSentences.foreach {
        case (ans, x) =>
          val connectedVariables = getVariablesConnectedToParagraphSentence(ans)
          val allVars = connectedVariables :+ x
          val coeffs = Seq.fill(connectedVariables.length)(-1.0) :+ 1.0
          ilpSolver.addConsBasicLinear("activeOptionVar", allVars, coeffs, None, Some(0.0))
          connectedVariables.foreach { connectedVar =>
            val vars = Seq(connectedVar, x)
            val coeffs = Seq(1.0, -1.0)
            ilpSolver.addConsBasicLinear("activeParagraphConsVar", vars, coeffs, None, Some(0.0))
          }
      }

      // active questions cons
      val activeQuestionConstituents = for {
        t <- qTokens
        weight = if (SolverUtils.scienceTermsMap.contains(t.getSurfaceForm.toLowerCase)) {
          params.activeQuestionTermWeight + params.scieneTermBoost
        } else {
          params.activeQuestionTermWeight
        }
        x = ilpSolver.createBinaryVar("activeQuestionCons", weight)
      } yield (t, x)
      // the question token is active if anything connected to it is active
      activeQuestionConstituents.foreach {
        case (c, x) =>
          val connectedVariables = getVariablesConnectedToQuestionToken(c)
          val allVars = connectedVariables :+ x
          val coeffs = Seq.fill(connectedVariables.length)(-1.0) :+ 1.0
          ilpSolver.addConsBasicLinear("activeQuestionIneq1", allVars, coeffs, None, Some(0.0))
          connectedVariables.foreach { connectedVar =>
            val vars = Seq(connectedVar, x)
            val coeffs = Seq(1.0, -1.0)
            ilpSolver.addConsBasicLinear("activeQuestionIneq2", vars, coeffs, None, Some(0.0))
          }
      }


      // extra weight for alignment of paragraph constituents
      // create edges between constituents which have an edge in the dependency parse
      // this edge can be active only if the base nodes are active
      def twoAnswerConsAreConnectedViaDependencyParse(ansIdx: Int, tokIdx1: Int, tokIdx2: Int): Boolean = {
        val cons1 = getAnswerOptionCons(ansIdx, tokIdx1)
        val cons2 = getAnswerOptionCons(ansIdx, tokIdx2)
        if (q.answers(ansIdx).aTAOpt.get.hasView(ViewNames.DEPENDENCY_STANFORD)) {
          val ansDepView = q.answers(ansIdx).aTAOpt.get.getView(ViewNames.DEPENDENCY_STANFORD)
          val cons1InDep = ansDepView.getConstituentsCovering(cons1).asScala.headOption
          val cons2InDep = ansDepView.getConstituentsCovering(cons2).asScala.headOption
          if (cons1InDep.isDefined && cons2InDep.isDefined) {
            val relations = ansDepView.getRelations.asScala
            relations.exists { r =>
              (r.getSource == cons1InDep.get && r.getTarget == cons2InDep.get) ||
                (r.getSource == cons2InDep.get && r.getTarget == cons1InDep.get)
            }
          }
          else {
            false
          }
        }
        else {
          false
        }
      }

      if (p.contextTAOpt.get.hasView(ViewNames.DEPENDENCY_STANFORD)) {
        val depView = p.contextTAOpt.get.getView(ViewNames.DEPENDENCY_STANFORD)
        val depRelations = depView.getRelations.asScala
        interParagraphAlignments = depRelations.zipWithIndex.map { case (r, idx) =>
          val startConsOpt = getParagraphConsCovering(r.getSource)
          val targetConsOpt = getParagraphConsCovering(r.getTarget)
          if (startConsOpt.isDefined && targetConsOpt.isDefined && startConsOpt.get != targetConsOpt.get) {
            val x = ilpSolver.createBinaryVar(s"Relation:$idx", params.firstOrderDependencyEdgeAlignments)

            // this relation variable is active, only if its two sides are active
            val startVar = activeParagraphConstituents(startConsOpt.get)
            val targetVar = activeParagraphConstituents(targetConsOpt.get)
            ilpSolver.addConsBasicLinear("dependencyVariableActiveOnlyIfSourceConsIsActive",
              Seq(x, startVar), Seq(1.0, -1.0), None, Some(0.0))
            ilpSolver.addConsBasicLinear("dependencyVariableActiveOnlyIfSourceConsIsActive",
              Seq(x, targetVar), Seq(1.0, -1.0), None, Some(0.0))

            val ansList1 = getAnswerOptionVariablesConnectedToParagraph(startConsOpt.get)
            val ansList2 = getAnswerOptionVariablesConnectedToParagraph(targetConsOpt.get)

            val variablesPairsInAnswerOptionsWithDependencyRelation = for {
              a <- ansList1
              b <- ansList2
              if a._1 == b._1 // same answer
              if a._2 != b._2 // different tok
              if twoAnswerConsAreConnectedViaDependencyParse(a._1, a._2, b._2) // they are connected via dep parse
            }
              yield {
                val weight = 0.0
                // TODO: tune this
                val activePair = ilpSolver.createBinaryVar(s"activeAnsweOptionPairs", weight)
                ilpSolver.addConsBasicLinear("NoActivePairIfNonAreActive", Seq(a._3, b._3, activePair), Seq(-1.0, -1.0, 1.0), None, Some(0.0))
                ilpSolver.addConsBasicLinear("NoActivePairIfNonAreActive", Seq(activePair, a._3), Seq(-1.0, 1.0), None, Some(0.0))
                ilpSolver.addConsBasicLinear("NoActivePairIfNonAreActive", Seq(activePair, b._3), Seq(-1.0, 1.0), None, Some(0.0))
                activePair
              }

            // if the paragraph relation pair is active, at least one answer response pair should be active
            // in other words
            ilpSolver.addConsBasicLinear("atLeastOnePairShouldBeActive",
              variablesPairsInAnswerOptionsWithDependencyRelation :+ x,
              Array.fill(variablesPairsInAnswerOptionsWithDependencyRelation.length) {
                -1.0
              } :+ 1.0, None, Some(0.0))

            Some(startConsOpt.get, targetConsOpt.get, x)
          }
          else {
            None
          }
        }.collect { case a if a.isDefined => a.get }
      }
      else {
        println("Paragraph does not contain parse-view . . . ")
      }

      // for each of the answer options create one variable, turning on when the number of the alignments to that answer
      // option is more than k
      /*    (1 to 3).foreach{ k: Int =>
          activeAnswerOptions.foreach { case (ansIdx, _) =>
          val penalty = k match {
            case 1  => params.moreThan1AlignmentAnsPenalty
            case 2  => params.moreThan2AlignmentAnsPenalty
            case 3  => params.moreThan3AlignmentAnsPenalty
          }
            val moreThanKAlignnetbToAnswerOption = ilpSolver.createBinaryVar(s"moreThan${k}AlignmentAnsPenalty", penalty)

            // this gets activated, if the answer option has at least two active alignments
            val connectedVariables = getVariablesConnectedToOption(ansIdx).toList
            val len = connectedVariables.length
            ilpSolver.addConsBasicLinear("",
              moreThanKAlignnetbToAnswerOption +: connectedVariables, (-len + k.toDouble) +: Array.fill(len) {1.0}, None, Some(k))
          }
        }*/


      /*    // for each of the question terms create one variable, turning on when the number of the alignments to the
        // constituent is more than k
        for{ k: Double <- 1.0 to 3.0 } {
          qTokens.foreach { c =>
            val penalty = k match {
              case 1.0  => params.moreThan1AlignmentToQuestionTermPenalty
              case 2.0  => params.moreThan2AlignmentToQuestionTermPenalty
              case 3.0  => params.moreThan3AlignmentToQuestionTermPenalty
            }
            val moreThanKAlignnetbToQuestionCons = ilpSolver.createBinaryVar(
              s"moreThan${k}AlignmentToQuestionConsPenalty", penalty
            )

            // this gets activated, if the answer option has at least two active alignments
            val connectedVariables = getVariablesConnectedToQuestionToken(c).toList
            val len = connectedVariables.length
            ilpSolver.addConsBasicLinear("",
              moreThanKAlignnetbToQuestionCons +: connectedVariables, (-len + k) +: Array.fill(len) {
                1.0
              }, None, Some(k))
          }
        }*/

      // there is an upper-bound on the max number of active tokens in each sentence
      activeSentences.foreach { case (ans, x) =>
        val connectedVariables = getVariablesConnectedToParagraphSentence(ans)
        ilpSolver.addConsBasicLinear("activeParagraphConsVar",
          connectedVariables, Array.fill(connectedVariables.length) {
            1.0
          },
          None, Some(params.maxNumberOfWordsAlignedPerSentence))
      }

      // among the words that are repeated in the paragraph, at most k of them can be active
      // first find the duplicate elements
      val duplicates = pTokens.groupBy(_.getSurfaceForm).filter { case (x, ys) => ys.lengthCompare(1) > 0 }
      duplicates.foreach { case (_, duplicateCons) =>
        val variables = duplicateCons.map(activeParagraphConstituents)
        ilpSolver.addConsBasicLinear("", variables, Array.fill(variables.length) {
          1.0
        },
          None, Some(params.maxAlignmentToRepeatedWordsInParagraph))
      }

      // have at most k active sentence
      val (_, sentenceVars) = activeSentences.unzip
      val sentenceVarsCoeffs = Seq.fill(sentenceVars.length)(1.0)
      ilpSolver.addConsBasicLinear("maxActiveParagraphConsVar", sentenceVars, sentenceVarsCoeffs,
        Some(0.0), Some(params.maxActiveSentences))


      // intra-sentence alignments
      // any sentences (that are at most k-sentences apart; k = 2 for now) can be aligned together.
      /*

        val maxIntraSentenceDistance = 2
        val intraSentenceAlignments = for{
          beginSentence <- 0 until (pTA.getNumberOfSentences - maxIntraSentenceDistance)
          offset <- 0 until maxIntraSentenceDistance
          endSentence = beginSentence + offset
          x = ilpSolver.createBinaryVar(s"interSentenceAlignment/$beginSentence/$endSentence", 0.0)
        } yield (beginSentence, endSentence, x)
        // co-reference
    */


      /*
        require(params.coreferenceWeight>=0, "params.coreferenceWeight should be positive")
        val corefCons = if (pTA.hasView(ViewNames.COREF)) pTA.getView(ViewNames.COREF).getConstituents.asScala else Seq.empty
        corefCons.groupBy(_.getLabel).foreach{ case (_, cons) =>  // cons that have the same label are co-refered
          cons.zipWithIndex.combinations(2).foreach{ consPair =>
            val x = ilpSolver.createBinaryVar(s"coredEdgeVariable${consPair.head._2}-${consPair(1)._2}", params.coreferenceWeight)
            val x1 = activeParagraphConstituents(consPair.head._1)
            val x2 = activeParagraphConstituents(consPair(1)._1)
            ilpSolver.addConsBasicLinear(s"coreEdgePairCons-${consPair.head._2}", Seq(x, x1), Seq(1.0, 1.0), None, Some(0.0))
            ilpSolver.addConsBasicLinear(s"coreEdgePairCons-${consPair(1)._2}", Seq(x, x2), Seq(1.0, 1.0), None, Some(0.0))
          }
        }
    */

      // longer than 1 answer penalty
      activeAnswerOptions.foreach { case (ansIdx, activeAnsVar) =>
        val ansTokList = aTokens(ansIdx)
        if (ansTokList.length > 1) {
          val x = ilpSolver.createBinaryVar("longerThanOnePenalty", params.longerThan1TokenAnsPenalty)
          ilpSolver.addConsBasicLinear("longerThanOnePenaltyActiveOnlyWhenOptionIsActive",
            Seq(x, activeAnsVar), Seq(-1.0, 1.0), None, Some(0.0))
        }
        if (ansTokList.length > 2) {
          val x = ilpSolver.createBinaryVar("longerThanTwoPenalty", params.longerThan2TokenAnsPenalty)
          ilpSolver.addConsBasicLinear("longerThanOnePenaltyActiveOnlyWhenOptionIsActive",
            Seq(x, activeAnsVar), Seq(-1.0, 1.0), None, Some(0.0))
        }
        if (ansTokList.length > 3) {
          val x = ilpSolver.createBinaryVar("longerThanThreePenalty", params.longerThan3TokenAnsPenalty)
          ilpSolver.addConsBasicLinear("longerThanThreePenaltyActiveOnlyWhenOptionIsActive",
            Seq(x, activeAnsVar), Seq(-1.0, 1.0), None, Some(0.0))
        }
      }

      // use at least k constituents in the question
      val (_, questionVars) = activeQuestionConstituents.unzip
      val questionVarsCoeffs = Seq.fill(questionVars.length)(1.0)
      ilpSolver.addConsBasicLinear("activeQuestionConsVarNum", questionVars,
        questionVarsCoeffs, Some(params.minQuestionTermsAligned), Some(params.maxQuestionTermsAligned))
      ilpSolver.addConsBasicLinear("activeQuestionConsVarRatio", questionVars,
        questionVarsCoeffs,
        Some(params.minQuestionTermsAlignedRatio * questionVars.length),
        Some(params.maxQuestionTermsAlignedRatio * questionVars.length))

      // if anything comes after " without " it should be aligned definitely
      // example: What would happen without annealing?
      if (q.questionText.contains(" without ")) {
        if (verbose) println(" >>> Adding constraint to use the term after `without`")
        val withoutTok = qTokens.filter(_.getSurfaceForm == "without").head
        if (verbose) println("withoutTok: " + withoutTok)
        val after = qTokens.filter(c => c.getStartSpan > withoutTok.getStartSpan).minBy(_.getStartSpan)
        if (verbose) println("after: " + after)
        val afterVariableOpt = activeQuestionConstituents.collectFirst { case (c, v) if c == after => v }
        if (verbose) println("afterVariableOpt = " + afterVariableOpt)
        afterVariableOpt match {
          case Some(afterVariable) =>
            ilpSolver.addConsBasicLinear("termAfterWithoutMustBeAligned", Seq(afterVariable), Seq(1.0), Some(1.0), None)
          case None => // do nothing
        }
      }
    }


    // constraint: answer option must be active if anything connected to it is active
    activeAnswerOptions.foreach {
      case (ansIdx, x) =>
        val connectedVariables = getVariablesConnectedToOption(ansIdx)
        val allVars = connectedVariables :+ x
        val coeffs = Seq.fill(connectedVariables.length)(-1.0) :+ 1.0
        ilpSolver.addConsBasicLinear("activeOptionVarImplesOneActiveConnectedEdge", allVars, coeffs, None, Some(0.0))
        connectedVariables.foreach { connectedVar =>
          val vars = Seq(connectedVar, x)
          val coeffs = Seq(1.0, -1.0)
          ilpSolver.addConsBasicLinear("activeConnectedEdgeImpliesOneAnswerOption", vars, coeffs, None, Some(0.0))
        }
    }

    // constraint: alignment to only one option, i.e. there must be only one single active option
    if (activeAnswerOptions.nonEmpty /*&& activeConstaints*/ ) {
      val activeAnsVars = activeAnswerOptions.map { case (ans, x) => x }
      val activeAnsVarsCoeffs = Seq.fill(activeAnsVars.length)(1.0)
      ilpSolver.addConsBasicLinear("onlyOneActiveOption", activeAnsVars, activeAnsVarsCoeffs, Some(1.0), Some(1.0))
    }

    // active answer option token
    /*
    val activeAnsweOptionToken = if(!isTrueFalseQuestion) {
      for {
        ansIdx <- aTokens.indices
        ansTokIdx <- aTokens(ansIdx).indices
        x = ilpSolver.createBinaryVar(s"activeAns${ansIdx}Tok${ansTokIdx}OptId", 0.0)
      } yield (ansIdx, ansTokIdx, x)
    }
    else {
      List.empty
    }
    */
    (questionParagraphAlignments, paragraphAnswerAlignments, interParagraphAlignments, activeAnswerOptions, isTrueFalseQuestion, aTokens)
  }

  def solveTopAnswer[V <: IlpVar](
                                   q: Question,
                                   p: Paragraph,
                                   ilpSolver: IlpSolver[V, _],
                                   alignmentFunction: AlignmentFunction,
                                   reasoningTypes: Set[ReasoningType],
                                   useSummary: Boolean,
                                   srlViewName: String
                                 ): (Seq[Int], EntityRelationResult) = {

    val modelCreationStart = System.currentTimeMillis()

    val (questionParagraphAlignments, paragraphAnswerAlignments, interParagraphAlignments, activeAnswerOptions, isTrueFalseQuestion, aTokens) = createILPModel[V](q, p, ilpSolver, alignmentFunction, reasoningTypes, useSummary, srlViewName)

    if (verbose) println("created the ilp model. Now solving it  . . . ")

    val modelSolveStart = System.currentTimeMillis()

    val numberOfBinaryVars = ilpSolver.getNOrigBinVars
    val numberOfContinuousVars = ilpSolver.getNOrigContVars
    val numberOfIntegerVars = ilpSolver.getNOrigIntVars
    val numberOfConstraints = ilpSolver.getNOrigConss

    // solving and extracting the answer
    ilpSolver.solve()

    val modelSolveEnd = System.currentTimeMillis()

    val statistics = Stats(numberOfBinaryVars, numberOfContinuousVars, numberOfIntegerVars, numberOfConstraints,
      ilpIterations = ilpSolver.getNLPIterations, modelCreationInSec = (modelSolveEnd - modelSolveStart) / 1000.0,
      solveTimeInSec = (modelSolveStart - modelCreationStart) / 1000.0, objectiveValue = ilpSolver.getPrimalbound)
    if (verbose) {
      println("Statistics: " + statistics)
      println("ilpSolver.getPrimalbound: " + ilpSolver.getPrimalbound)
    }

    if (verbose) println("Done solving the model  . . . ")

    if (verbose) println("paragraphAnswerAlignments: " + paragraphAnswerAlignments.length)

    def getVariablesConnectedToOptionToken(ansIdx: Int, tokenIdx: Int): Seq[V] = {
      paragraphAnswerAlignments.filter { case (_, ansIdxTmp, tokenIdxTmp, _) =>
        ansIdxTmp == ansIdx && tokenIdxTmp == tokenIdx
      }.map(_._4)
    }

    def getVariablesConnectedToOption(ansIdx: Int): Seq[V] = {
      paragraphAnswerAlignments.filter { case (_, ansTmp, _, _) => ansTmp == ansIdx }.map(_._4)
    }

    def stringifyVariableSequence(seq: Seq[(Int, V)]): String = {
      seq.map { case (id, x) => "id: " + id + " : " + ilpSolver.getSolVal(x) }.mkString(" / ")
    }

    def stringifyVariableSequence3(seq: Seq[(Constituent, V)])(implicit d: DummyImplicit): String = {
      seq.map { case (id, x) => "id: " + id.getSurfaceForm + " : " + ilpSolver.getSolVal(x) }.mkString(" / ")
    }

    def stringifyVariableSequence2(seq: Seq[(Constituent, Constituent, V)])(implicit d: DummyImplicit, d2: DummyImplicit): String = {
      seq.map { case (c1, c2, x) => "c1: " + c1.getSurfaceForm + ", c2: " + c2.getSurfaceForm + " -> " + ilpSolver.getSolVal(x) }.mkString(" / ")
    }

    def stringifyVariableSequence4(seq: Seq[(Constituent, Int, Int, V)]): String = {
      seq.map { case (c, i, j, x) => "c: " + c.getSurfaceForm + ", ansIdx: " + i + ", ansConsIdx: " + j + " -> " + ilpSolver.getSolVal(x) }.mkString(" / ")
    }

    if (ilpSolver.getStatus == IlpStatusOptimal) {
      if (verbose) println("Primal score: " + ilpSolver.getPrimalbound)
      val trueIdx = q.trueIndex
      val falseIdx = q.falseIndex
      val selectedIndex = getSelectedIndices(ilpSolver, activeAnswerOptions, isTrueFalseQuestion, trueIdx, falseIdx)
      val questionBeginning = "Question: "
      val paragraphBeginning = "|Paragraph: "
      val questionString = questionBeginning + q.questionText
      val choiceString = "|Options: " + q.answers.zipWithIndex.map { case (ans, key) => s" (${key + 1}) " + ans.answerText }.mkString(" ")
      val paragraphString = paragraphBeginning + p.context

      val entities = ArrayBuffer[Entity]()
      val relations = ArrayBuffer[Relation]()
      var eIter = 0
      var rIter = 0

      val entityMap = scala.collection.mutable.Map[(Int, Int), String]()
      val relationSet = scala.collection.mutable.Set[(String, String)]()

      questionParagraphAlignments.foreach {
        case (c1, c2, x) =>
          if (ilpSolver.getSolVal(x) > 1.0 - TextILPSolver.epsilon) {
            val qBeginIndex = questionBeginning.length + c1.getStartCharOffset
            val qEndIndex = qBeginIndex + c1.getSurfaceForm.length
            val span1 = (qBeginIndex, qEndIndex)
            val t1 = if (!entityMap.contains(span1)) {
              val t1 = "T" + eIter
              eIter = eIter + 1
              entities += Entity(t1, c1.getSurfaceForm, Seq(span1))
              entityMap.put(span1, t1)
              t1
            }
            else {
              entityMap(span1)
            }
            val pBeginIndex = c2.getStartCharOffset + questionString.length + paragraphBeginning.length
            val pEndIndex = pBeginIndex + c2.getSurfaceForm.length
            val span2 = (pBeginIndex, pEndIndex)
            val t2 = if (!entityMap.contains(span2)) {
              val t2 = "T" + eIter
              eIter = eIter + 1
              entities += Entity(t2, c2.getSurfaceForm, Seq(span2))
              entityMap.put(span2, t2)
              t2
            }
            else {
              entityMap(span2)
            }

            if (!relationSet.contains((t1, t2))) {
              relations += Relation("R" + rIter, t1, t2, ilpSolver.getVarObjCoeff(x))
              rIter = rIter + 1
              relationSet.add((t1, t2))
            }
          }
      }

      paragraphAnswerAlignments.foreach {
        case (c1, ansIdx, ansConsIdx, x) =>
          if (ilpSolver.getSolVal(x) > 1.0 - TextILPSolver.epsilon) {
            val pBeginIndex = c1.getStartCharOffset + questionString.length + paragraphBeginning.length
            val pEndIndex = pBeginIndex + c1.getSurfaceForm.length
            val span1 = (pBeginIndex, pEndIndex)
            val t1 = if (!entityMap.contains(span1)) {
              val t1 = "T" + eIter
              entities += Entity(t1, c1.getSurfaceForm, Seq(span1))
              entityMap.put(span1, t1)
              eIter = eIter + 1
              t1
            } else {
              entityMap(span1)
            }

            val ansString = aTokens(ansIdx)(ansConsIdx)
            val ansswerBeginIdx = choiceString.indexOf(q.answers(ansIdx).answerText)
            val oBeginIndex = choiceString.indexOf(ansString, ansswerBeginIdx) + questionString.length + paragraphString.length
            val oEndIndex = oBeginIndex + ansString.length
            val span2 = (oBeginIndex, oEndIndex)
            val t2 = if (!entityMap.contains(span2)) {
              val t2 = "T" + eIter
              entities += Entity(t2, ansString, Seq(span2))
              eIter = eIter + 1
              entityMap.put(span2, t2)
              t2
            }
            else {
              entityMap(span2)
            }
            if (!relationSet.contains((t1, t2))) {
              relations += Relation("R" + rIter, t1, t2, ilpSolver.getVarObjCoeff(x))
              rIter = rIter + 1
              relationSet.add((t1, t2))
            }
          }
      }

      // inter-paragraph alignments
      interParagraphAlignments.foreach { case (c1, c2, x) =>
        if (ilpSolver.getSolVal(x) > 1.0 - TextILPSolver.epsilon) {
          val pBeginIndex1 = c1.getStartCharOffset + questionString.length + paragraphBeginning.length
          val pEndIndex1 = pBeginIndex1 + c1.getSurfaceForm.length
          val span1 = (pBeginIndex1, pEndIndex1)
          val t1 = if (!entityMap.contains(span1)) {
            val t1 = "T" + eIter
            entities += Entity(t1, c1.getSurfaceForm, Seq(span1))
            entityMap.put(span1, t1)
            eIter = eIter + 1
            t1
          } else {
            entityMap(span1)
          }
          val pBeginIndex2 = c2.getStartCharOffset + questionString.length + paragraphBeginning.length
          val pEndIndex2 = pBeginIndex2 + c2.getSurfaceForm.length
          val span2 = (pBeginIndex2, pEndIndex2)
          val t2 = if (!entityMap.contains(span2)) {
            val t2 = "T" + eIter
            entities += Entity(t2, c2.getSurfaceForm, Seq(span2))
            entityMap.put(span2, t2)
            eIter = eIter + 1
            t2
          } else {
            entityMap(span2)
          }
          if (!relationSet.contains((t1, t2))) {
            relations += Relation("R" + rIter, t1, t2, ilpSolver.getVarObjCoeff(x))
            rIter = rIter + 1
            relationSet.add((t1, t2))
          }
        }

        if (isTrueFalseQuestion) {
          // add the answer option span manually
          selectedIndex.foreach { idx =>
            val ansText = q.answers(idx).answerText
            val oBeginIndex = choiceString.indexOf(ansText) + questionString.length + paragraphString.length
            val oEndIndex = oBeginIndex + ansText.length
            val span2 = (oBeginIndex, oEndIndex)
            val t2 = if (!entityMap.contains(span2)) {
              val t2 = "T" + eIter
              entities += Entity(t2, ansText, Seq(span2))
              eIter = eIter + 1
              entityMap.put(span2, t2)
              t2
            }
            else {
              entityMap(span2)
            }
          }
        }
      }

      if (verbose) println("returning the answer  . . . ")

      val solvedAnswerLog = "activeAnswerOptions: " + stringifyVariableSequence(activeAnswerOptions) +
        //"  activeQuestionConstituents: " + stringifyVariableSequence3(activeQuestionConstituents) +
        "  questionParagraphAlignments: " + stringifyVariableSequence2(questionParagraphAlignments) +
        "  paragraphAnswerAlignments: " + stringifyVariableSequence4(paragraphAnswerAlignments) +
        "  aTokens: " + aTokens.toString

      val erView = EntityRelationResult(questionString + paragraphString + choiceString, entities, relations,
        confidence = ilpSolver.getPrimalbound, log = solvedAnswerLog, statistics = statistics)
      selectedIndex -> erView
    }
    else {
      if (verbose) println("Not optimal . . . ")
      if (verbose) println("Status is not optimal. Status: " + ilpSolver.getStatus)
      // if the program is not solver, say IDK
      Seq.empty -> EntityRelationResult("INFEASIBLE: " + reasoningTypes, List.empty, List.empty, statistics = statistics)
    }
  }


  private def getSelectedIndices[V <: IlpVar](ilpSolver: IlpSolver[V, _], activeAnswerOptions: Seq[(Int, V)], isTrueFalseQuestion: Boolean, trueIdx: Int, falseIdx: Int) = {
    if (isTrueFalseQuestion) {
      if (ilpSolver.getPrimalbound > TextILPSolver.trueFalseThreshold) Seq(trueIdx) else Seq(falseIdx)
    }
    else {
      if (verbose) println(">>>>>>> not true/false . .. ")
      activeAnswerOptions.zipWithIndex.collect { case ((ans, x), idx) if ilpSolver.getSolVal(x) > 1.0 - TextILPSolver.epsilon => idx }
    }
  }

}
