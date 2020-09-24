package org.allenai.ari.solvers.textilp.solvers

import java.io.{BufferedWriter, File, FileWriter}
import java.net.URL

import edu.cmu.meteor.scorer.{MeteorConfiguration, MeteorScorer}
import edu.illinois.cs.cogcomp.core.datastructures.ViewNames
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.{Constituent, PredicateArgumentView, TextAnnotation}
import org.allenai.ari.controller.questionparser.{FillInTheBlankGenerator, QuestionParse}
import org.simmetrics.StringMetric
import org.simmetrics.metrics.StringMetrics

import scala.collection.mutable
import org.allenai.ari.solvers.bioProccess.ProcessBankReader._
import org.allenai.ari.solvers.textilp.{EntityRelationResult, _}
import org.allenai.ari.solvers.textilp.alignment.{AlignmentFunction, KeywordTokenizer}
import org.allenai.ari.solvers.textilp.ilpsolver.{IlpVar, _}
import org.allenai.ari.solvers.textilp.solvers.TextILPSolver.minPConsToPConsAlignment
import org.allenai.ari.solvers.textilp.utils.{AnnotationUtils, Constants, SolverUtils}

import scala.collection.JavaConverters._
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

trait ReasoningType {}

case object SimpleMatching extends ReasoningType

case object SimpleMatchingWithCoref extends ReasoningType

case object SRLV1ILP extends ReasoningType

case object SRLV1Rule extends ReasoningType

case object SRLV2Rule extends ReasoningType

case object SRLV3Rule extends ReasoningType

case object VerbSRLandCommaSRL extends ReasoningType

case object VerbSRLandCoref extends ReasoningType

case object VerbSRLandPrepSRL extends ReasoningType

case object CauseRule extends ReasoningType

case object WhatDoesItDoRule extends ReasoningType

trait TextILPModel {}

object TextILPModel {
  case object MyModel extends TextILPModel
  // ensemble of annotators; this achieves good results across the two datasets; used in AAAI paper
  case object EnsembleFull extends TextILPModel

  // ensemble of annotators; this achieves good results across the two datasets
  case object EnsembleMinimal extends TextILPModel

  // stacked version; acheives good (and fast) results on science exams
  case object StackedForScience extends TextILPModel

  // stacked version; acheives good (and fast) results on science exams; no curator annotations
  case object StackedForScienceMinimal extends TextILPModel

  // stacked version; acehieves good (and fast) results on process-bank data
  case object StackedForProcesses extends TextILPModel

  // stacked version; acehieves good (and fast) results on process-bank data; no curator annotations
  case object StackedForProcessesMinimal extends TextILPModel


  // created for ablation study
  case object EnsembleNoSimpleMatching extends TextILPModel

  case object EnsembleNoVerbSRL extends TextILPModel

  case object EnsembleNoCoref extends TextILPModel

  case object EnsembleNoCommaSRL extends TextILPModel

  case object EnsembleNoNomSRL extends TextILPModel

  case object EnsembleNoPrepSRL extends TextILPModel

}

object TextILPSolver {
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
  //val minPConsToPConsAlignment = 0.6
  val minPConsToPConsAlignment = -0.4 //tuned for intersentence pcons alignment offset
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

  def getMaxScore(qCons: Seq[Constituent], pCons: Seq[Constituent]): Double = {
    qCons.flatMap { qC =>
      pCons.map { pC =>
        offlineAligner.scoreCellCell(pC.getSurfaceForm, qC.getSurfaceForm)
      }
    }.max
  }

  def getAvgScore(qCons: Seq[Constituent], pCons: Seq[Constituent]): Double = {
    val scoreList = qCons.flatMap { qC =>
      pCons.map { pC =>
        offlineAligner.scoreCellCell(pC.getSurfaceForm, qC.getSurfaceForm)
      }
    }
    scoreList.sum / scoreList.length
  }

  lazy val offlineAligner = new AlignmentFunction("Entailment", 0.2,
    TextILPSolver.keywordTokenizer, useRedisCache = false, useContextInRedisCache = false)


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

class TextILPSolver(annotationUtils: AnnotationUtils,
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

  def solveWithReasoningType(question: String, options: Seq[String], snippet: String, reasoningType: ReasoningType, srlViewName: String = TextILPSolver.pathLSTMViewName): (Seq[Int], EntityRelationResult) = {
    val (q: Question, p: Paragraph) = preprocessQuestionData(question, options, snippet)
    reasoningType match {

      case x =>
        val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
        val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(x), useSummary = true, srlViewName)
        ilpSolver.free()
        result
    }
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
                                 ): (mutable.Buffer[(Constituent, Constituent, V)], mutable.Buffer[(Constituent, Int, Int, V)], mutable.Buffer[(Constituent, Constituent, V)], Seq[(Int, V)], Boolean, Seq[Seq[String]],mutable.Buffer[(Int,V)], mutable.Buffer[(Constituent, Constituent, V)]) = {
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
    val activeSentences: mutable.Buffer[(Int,V)] = mutable.Buffer.empty
    var interSentenceAlignments : mutable.Buffer[(Constituent,Constituent,V)] = mutable.Buffer.empty
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

      val interSentenceTokenAlignments = for {
        pCons1 <- pTokens
        pCons2 <- pTokens
        score = alignmentFunction.scoreCellCell(pCons1.getSurfaceForm, pCons2.getSurfaceForm) + minPConsToPConsAlignment
        if (score > 0.0)
        if (pCons1.getSentenceId != pCons2.getSentenceId)
           x = ilpSolver.createBinaryVar(name = "", score)

      } yield (pCons1,pCons2,x)

      interSentenceAlignments ++= interSentenceTokenAlignments.toBuffer

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
          paragraphTokenAnswerAlignments.filter { case (cTmp, _, _, _) => cTmp == c }.map(_._4) ++
          interSentenceTokenAlignments.filter{ case (cTmp,_,_) => cTmp==c }.map(_._3)
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
      val activeSent = for {
        s <- 0 until pTA.getNumberOfSentences
        // alignment is preferred for lesser sentences; hence: negative activeSentenceDiscount
        x = ilpSolver.createBinaryVar("activeSentence:" + s, params.activeSentencesDiscount)
      } yield (s, x)
      activeSentences ++= activeSent.toBuffer

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

    
    (questionParagraphAlignments, paragraphAnswerAlignments, interParagraphAlignments, activeAnswerOptions, isTrueFalseQuestion, aTokens, activeSentences, interSentenceAlignments)
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

    val (questionParagraphAlignments, paragraphAnswerAlignments, interParagraphAlignments, activeAnswerOptions, isTrueFalseQuestion, aTokens, activeSentences, interSentenceAlignments) = createILPModel[V](q, p, ilpSolver, alignmentFunction, reasoningTypes, useSummary, srlViewName)

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
    //ilpSolver.exportModel("Orig_ILP_Model",true)
    //ilpSolver.exportModel("Reduced_ILP_Model",false)
    val iter = ilpSolver.getAllActiveVars
    //ilpSolver.printResult(iter)
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
      var sentList = new ListBuffer[Int]()

      activeSentences.foreach{
        case(x,s) =>
        if (ilpSolver.getSolVal(s) > 1.0 - TextILPSolver.epsilon) 
        { sentList += x}
      }
      val pTA = p.contextTAOpt.getOrElse(throw new Exception("The annotation for the paragraph not found . . . "))
      val sentences = pTA.getNumberOfSentences
      //val sentences = p.context.split(" . ")
      //val activeSentList = sentList.map(sentences(_)).mkString(",")
      val sentindexes = (0 to sentences-1).toList

      val qpascores = scala.collection.mutable.MutableList.empty[Double]

      sentindexes.foreach{ sentid=>
        var Entscore=0.0
      questionParagraphAlignments.foreach {
        case (c1, c2, x) =>
          if (c2.getSentenceId==sentid)
          if (ilpSolver.getSolVal(x) > 1.0 - TextILPSolver.epsilon)
            Entscore += ilpSolver.getVarObjCoeff(x)
      }
        qpascores +=Entscore
      }


      val paascores = scala.collection.mutable.MutableList.empty[Double]

      sentindexes.foreach { sentid =>
        var Entscore2=0.0
        paragraphAnswerAlignments.foreach {
          case (c1, a1, a2, x) =>
            if (c1.getSentenceId == sentid)
            if (ilpSolver.getSolVal(x) > 1.0 - TextILPSolver.epsilon)
              Entscore2 += ilpSolver.getVarObjCoeff(x)
        }
        paascores +=Entscore2
      }


      val interParaScores= scala.collection.mutable.MutableList.empty[Double]

      sentindexes.foreach{ sentid=>
        var ipscore=0.0
      interParagraphAlignments.foreach {
        case (c1, c2, x) =>
          if (c1.getSentenceId == sentid)
            if (ilpSolver.getSolVal(x) > 1.0 - TextILPSolver.epsilon)
              ipscore += ilpSolver.getVarObjCoeff(x)
      }
          interParaScores+=ipscore
      }


      val interSentScores = scala.collection.mutable.MutableList.empty[Double]

      sentindexes.foreach { sentID=>
        var sentScore = 0.0
        interSentenceAlignments.foreach { case (c1, c2, x) =>
         if (c1.getSentenceId == sentID) {
          if (ilpSolver.getSolVal(x) > 1.0 - TextILPSolver.epsilon) {
            sentScore += ilpSolver.getVarObjCoeff(x)
          }
         }
         }
        interSentScores += (sentScore/2)
        }

      val finalSentScores = List(qpascores, paascores, interParaScores, interSentScores).transpose.map(_.sum)
      //val finalSentScores = (qpascores, paascores,interParaScores).zipped.map(_ + _ + _)
      val zippedSenScores =(sentindexes zip finalSentScores).toMap
      val sortedMap = scala.collection.immutable.ListMap(zippedSenScores.toSeq.sortWith(_._2 > _._2):_*)
      val filename="AlignmentScores4.tsv"
      val file = new File(filename)
      val bw = new BufferedWriter(new FileWriter(file,true))
      for (i<- 0 to sentindexes.length-1 ) {
        val line= q.questionText +"\t"+ sentindexes(i) + "\t" + qpascores(i).toString+"\t"+paascores(i).toString+interParaScores(i).toString+"\t"+interSentScores+"\n"
        bw.write(line)
      }
      bw.close()

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
      val solvedAnswerLog = sortedMap.toString
      /*val solvedAnswerLog = "activeAnswerOptions: " + stringifyVariableSequence(activeAnswerOptions) +
        "  interParagraphAlignments: " + stringifyVariableSequence2(interParagraphAlignments) +
        "  questionParagraphAlignments: " + stringifyVariableSequence2(questionParagraphAlignments) +
        "  paragraphAnswerAlignments: " + stringifyVariableSequence4(paragraphAnswerAlignments) +
        " activeSentenceID: " + stringifyVariableSequence(activeSentences) +
        " activeSentences: " + activeSentList +
        "  aTokens: " + aTokens.toString +
        " scoredSentences: " + sortedMap.toString +
        "interParaScores: " + listofscores3 +
        "interSentScores:" + interSentScores*/


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

  def solveAllAnswerOptions(question: String, options: Seq[String], snippet: String, reasoningType: ReasoningType, srlVu: String): Seq[(Seq[Int], EntityRelationResult)] = {
    val annotationStart = System.currentTimeMillis()
    val (q: Question, p: Paragraph) = preprocessQuestionData(question, options, snippet)
    val annotationEnd = System.currentTimeMillis()

    reasoningType match {

      case x =>
        val ilpSolver = new ScipSolver("textILP", ScipParams.Default)

        val modelCreationStart = System.currentTimeMillis()

        val (questionParagraphAlignments, paragraphAnswerAlignments,
        interParagraphAlignments, activeAnswerOptions,
        isTrueFalseQuestion, aTokens, activeSentences,interSentenceAlignments) = createILPModel(q, p, ilpSolver, aligner, Set(reasoningType), useSummary = true, srlVu)
        val modelCreationEnd = System.currentTimeMillis()

        if (verbose) println("created the ilp model. Now solving it  . . . ")

        val activeAnswerOptionsMap = activeAnswerOptions.toMap

        def solveExcludingAnswerOptions(toExclude: Set[Int],
                                        solutionsSoFar: Seq[(Seq[Int], EntityRelationResult)]): Seq[(Seq[Int], EntityRelationResult)] = {
          println(s"Disabling choice: $toExclude")
          toExclude.foreach { idx =>
            ilpSolver.chgVarUb(activeAnswerOptionsMap(idx), 0d)
          }

          val modelSolveStart = System.currentTimeMillis()

          val numberOfBinaryVars = ilpSolver.getNOrigBinVars
          val numberOfContinuousVars = ilpSolver.getNOrigContVars
          val numberOfIntegerVars = ilpSolver.getNOrigIntVars
          val numberOfConstraints = ilpSolver.getNOrigConss

          // solving and extracting the answer
          ilpSolver.solve()

          val modelSolveEnd = System.currentTimeMillis()

          val statistics = Stats(numberOfBinaryVars, numberOfContinuousVars, numberOfIntegerVars, numberOfConstraints,
            ilpIterations = ilpSolver.getNLPIterations, modelCreationInSec = (modelSolveEnd - modelCreationEnd) / 1000.0,
            solveTimeInSec = (modelSolveStart - modelSolveEnd) / 1000.0, objectiveValue = ilpSolver.getPrimalbound)
          if (verbose) {
            println("Statistics: " + statistics)
            println("ilpSolver.getPrimalbound: " + ilpSolver.getPrimalbound)
          }

          val newSolution = if (ilpSolver.getStatus == IlpStatusOptimal) {
            println("Primal score: " + ilpSolver.getPrimalbound)
            val trueIdx = q.trueIndex
            val falseIdx = q.falseIndex
            val selectedIndex = getSelectedIndices(ilpSolver, activeAnswerOptions, isTrueFalseQuestion, trueIdx, falseIdx)
            println("selectedIndex: " + selectedIndex)
            selectedIndex -> EntityRelationResult(snippet, List.empty, List.empty,
              statistics = statistics.copy(selected = if (toExclude.isEmpty) true else false))
          } else {
            println("Not optimal . . . ")
            if (verbose) println("Status is not optimal. Status: " + ilpSolver.getStatus)
            // if the program is not solver, say IDK
            Seq.empty -> EntityRelationResult("INFEASIBLE", List.empty, List.empty, statistics = statistics)
          }

          val newToExclude = toExclude ++ newSolution._1

          if (newToExclude.size < activeAnswerOptions.size && newSolution._1.nonEmpty) {
            // Reset solution for any future calls to solve
            ilpSolver.resetSolve()
            // continue solving : call the method again with the best choice disabled
            solveExcludingAnswerOptions(newToExclude, solutionsSoFar :+ newSolution)
          } else {
            solutionsSoFar :+ newSolution
          }
        }

        val result = solveExcludingAnswerOptions(Set.empty, Seq.empty)
        ilpSolver.free()
        result
    }
  }

}
