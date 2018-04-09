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

  lazy val sahandClient = new SahandClient(s"${Constants.sahandServer}:${Constants.sahandPort}/")

  val toBeVerbs = Set("am", "is", "are", "was", "were", "being", "been", "be", "were", "be")

  import weka.core.SerializationHelper

  lazy val wekaClassifier = Constants.textILPModel match {
    case TextILPModel.EnsembleFull => SerializationHelper.read("output/logistic.model").asInstanceOf[Classifier]
    case TextILPModel.EnsembleMinimal => SerializationHelper.read("output/logistic-nocurator2.model").asInstanceOf[Classifier]

    // created for ablation study
    case TextILPModel.EnsembleNoSimpleMatching => SerializationHelper.read("output/logistic-nosimplematching.model").asInstanceOf[Classifier]
    case TextILPModel.EnsembleNoPrepSRL => SerializationHelper.read("output/logistic-noprepsrl.model").asInstanceOf[Classifier]
    case TextILPModel.EnsembleNoVerbSRL => SerializationHelper.read("output/logistic-noverbsrl.model").asInstanceOf[Classifier]
    case TextILPModel.EnsembleNoCommaSRL => SerializationHelper.read("output/logistic-nocommasrl.model").asInstanceOf[Classifier]
    case TextILPModel.EnsembleNoCoref => SerializationHelper.read("output/logistic-nocoref.model").asInstanceOf[Classifier]
  }

  lazy val headerEmptyInstances = {
    val loader = new ArffLoader()
    Constants.textILPModel match {
      case TextILPModel.EnsembleFull => loader.setFile(new File("output/headerFile.arff"))
      case TextILPModel.EnsembleMinimal => loader.setFile(new File("output/headerFile_nocurator.arff"))

      // created for ablation
      case TextILPModel.EnsembleNoSimpleMatching => loader.setFile(new File("output/headerFile_nosimplematching.arff"))
      case TextILPModel.EnsembleNoPrepSRL => loader.setFile(new File("output/headerFile_noprepsrl.arff"))
      case TextILPModel.EnsembleNoVerbSRL => loader.setFile(new File("output/headerFile_noverbsrl.arff"))
      case TextILPModel.EnsembleNoCommaSRL => loader.setFile(new File("output/headerFile_nocomma.arff"))
      case TextILPModel.EnsembleNoCoref => loader.setFile(new File("output/headerFile-nocoref.arff"))
    }
    loader.getDataSet
  }
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

  def SRLSolverV1WithAllViews(q: Question, p: Paragraph): (Seq[Int], EntityRelationResult) = {
    lazy val srlVerbPipeline = SRLSolverV1(q, p, ViewNames.SRL_VERB)
    lazy val srlVerbCurator = SRLSolverV1(q, p, TextILPSolver.curatorSRLViewName)
    if (srlVerbPipeline._1.nonEmpty) {
      srlVerbPipeline
    }
    else if (srlVerbCurator._1.nonEmpty) {
      srlVerbCurator
    }
    else {
      SRLSolverV1(q, p, TextILPSolver.pathLSTMViewName)
    }
  }

  def SRLSolverV2WithAllViews(q: Question, p: Paragraph): (Seq[Int], EntityRelationResult) = {
    lazy val srlVerbPipeline = SRLSolverV2(q, p, ViewNames.SRL_VERB)
    lazy val srlVerbCurator = SRLSolverV2(q, p, TextILPSolver.curatorSRLViewName)
    if (srlVerbPipeline._1.nonEmpty) {
      srlVerbPipeline
    }
    else if (srlVerbCurator._1.nonEmpty) {
      srlVerbCurator
    }
    else {
      SRLSolverV2(q, p, TextILPSolver.pathLSTMViewName)
    }
  }

  def SRLSolverV3WithAllViews(q: Question, p: Paragraph, alignmentFunction: AlignmentFunction): (Seq[Int], EntityRelationResult) = {
    lazy val srlVerbPipeline = SRLSolverV3(q, p, alignmentFunction, ViewNames.SRL_VERB)
    lazy val srlVerbCurator = try {
      // because curator annotation stuff fail sometime
      SRLSolverV3(q, p, alignmentFunction, TextILPSolver.curatorSRLViewName)
    }
    catch {
      case e: Exception => e.printStackTrace()
        SRLSolverV3(q, p, alignmentFunction, TextILPSolver.pathLSTMViewName)
    }

    if (srlVerbPipeline._1.nonEmpty) {
      srlVerbPipeline
    }
    else if (srlVerbCurator._1.nonEmpty) {
      srlVerbCurator
    }
    else {
      SRLSolverV3(q, p, alignmentFunction, TextILPSolver.pathLSTMViewName)
    }
  }

  def solve(question: String, options: Seq[String], snippet: String): (Seq[Int], EntityRelationResult) = {
    val (q: Question, p: Paragraph) = preprocessQuestionData(question, options, snippet)
    require(p.contextTAOpt.isDefined, "pTA is not defined after pre-processing . . . ")
    require(q.qTAOpt.isDefined, "qTA is not defined after pre-processing . . . ")
    println("Reasoning methods . . . ")
    lazy val resultSRLV1_path_lstm = SRLSolverV1(q, p, TextILPSolver.pathLSTMViewName) -> "resultSRLV1_path_lstm"
    lazy val resultSRLV1_curator = SRLSolverV1(q, p, TextILPSolver.curatorSRLViewName) -> "resultSRLV1_curator"
    lazy val resultSRLV1_pipeline = SRLSolverV1(q, p, ViewNames.SRL_VERB) -> "resultSRLV1_pipeline"

    lazy val resultSRLV2_path_lstm = SRLSolverV2(q, p, TextILPSolver.pathLSTMViewName) -> "resultSRLV2_path_lstm"
    lazy val resultSRLV2_curator = SRLSolverV2(q, p, TextILPSolver.curatorSRLViewName) -> "resultSRLV2_curator"
    lazy val resultSRLV2_pipeline = SRLSolverV2(q, p, ViewNames.SRL_VERB) -> "resultSRLV2_pipeline"

    lazy val resultSRLV3_path_lstm = SRLSolverV3(q, p, aligner, TextILPSolver.pathLSTMViewName) -> "resultSRLV3_path_lstm"
    lazy val resultSRLV3_curator = SRLSolverV3(q, p, aligner, TextILPSolver.curatorSRLViewName) -> "resultSRLV3_curator"
    lazy val resultSRLV3_pipeline = SRLSolverV3(q, p, aligner, ViewNames.SRL_VERB) -> "resultSRLV3_pipeline"

    lazy val resultWhatDoesItdo = WhatDoesItDoSolver(q, p) -> "resultWhatDoesItdo"
    lazy val resultCause = CauseResultRules(q, p) -> "resultCause"
    lazy val resultVerbSRLPlusCommaSRL_pipelneSRL = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(VerbSRLandCommaSRL), useSummary = true, ViewNames.SRL_VERB)
      ilpSolver.free()
      result
    } -> "resultVerbSRLPlusCommaSRL_pipeline_srl"
    lazy val resultVerbSRLPlusCommaSRL_curatorSRL = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(VerbSRLandCommaSRL), useSummary = true, TextILPSolver.curatorSRLViewName)
      ilpSolver.free()
      result
    } -> "resultVerbSRLPlusCommaSRL_curator_srl"
    lazy val resultVerbSRLPlusCommaSRL_pathLSTM = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(VerbSRLandCommaSRL), useSummary = true, TextILPSolver.pathLSTMViewName)
      ilpSolver.free()
      result
    } -> "resultVerbSRLPlusCommaSRL"
    lazy val resultSimpleMatching = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(SimpleMatching), useSummary = true, TextILPSolver.pathLSTMViewName)
      ilpSolver.free()
      result
    } -> "resultILP"
    lazy val resultVerbSRLPlusCoref_pipelineSRL = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(VerbSRLandCoref), useSummary = true, ViewNames.SRL_VERB)
      ilpSolver.free()
      result
    } -> "resultVerbSRLPlusCoref_pipeline_srl"
    lazy val resultVerbSRLPlusCoref_curatorSRL = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(VerbSRLandCoref), useSummary = true, TextILPSolver.curatorSRLViewName)
      ilpSolver.free()
      result
    } -> "resultVerbSRLPlusCoref_curator_srl"
    lazy val resultVerbSRLPlusCoref_pathLSTM = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(VerbSRLandCoref), useSummary = true, TextILPSolver.pathLSTMViewName)
      ilpSolver.free()
      result
    } -> "resultVerbSRLPlusCoref_path_lstm"
    lazy val resultVerbSRLPlusPrepSRL_pipeline_srl = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(VerbSRLandPrepSRL), useSummary = true, ViewNames.SRL_VERB)
      ilpSolver.free()
      result
    } -> "resultVerbSRLPlusPrepSRL_pipeline_srl"
    lazy val resultVerbSRLPlusPrepSRL_curator_srl = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(VerbSRLandPrepSRL), useSummary = true, TextILPSolver.curatorSRLViewName)
      ilpSolver.free()
      result
    } -> "resultVerbSRLPlusPrepSRL_curator_srl"
    lazy val resultVerbSRLPlusPrepSRL_path_lstm = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(VerbSRLandPrepSRL), useSummary = true, TextILPSolver.pathLSTMViewName)
      ilpSolver.free()
      result
    } -> "resultVerbSRLPlusPrepSRL_path_lstm"
    lazy val srlV1ILP_pipeline_srl = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(SRLV1ILP), useSummary = false, ViewNames.SRL_VERB)
      ilpSolver.free()
      result
    } -> "srlV1ILP_pipeline_srl"
    lazy val srlV1ILP_curator_srl = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(SRLV1ILP), useSummary = false, TextILPSolver.curatorSRLViewName)
      ilpSolver.free()
      result
    } -> "srlV1ILP_curator_srl"
    lazy val srlV1ILP_path_lstm = {
      val ilpSolver = new ScipSolver("textILP", ScipParams.Default)
      val result = solveTopAnswer(q, p, ilpSolver, aligner, Set(SRLV1ILP), useSummary = false, TextILPSolver.pathLSTMViewName)
      ilpSolver.free()
      result
    } -> "srlV1ILP_path_lstm"

    val resultOpt = Constants.textILPModel match {
      case TextILPModel.StackedForScience =>
/*
        (resultVerbSRLPlusCoref_curatorSRL #:: resultVerbSRLPlusCommaSRL_pathLSTM #:: resultSRLV3_curator #::
          resultVerbSRLPlusCommaSRL_curatorSRL #:: resultVerbSRLPlusCommaSRL_pipelneSRL #:: resultSRLV3_pipeline #::
          srlV1ILP_curator_srl #:: srlV1ILP_pipeline_srl #:: resultSRLV3_path_lstm #:: resultVerbSRLPlusCoref_pathLSTM #::
          resultVerbSRLPlusPrepSRL_path_lstm #:: resultVerbSRLPlusPrepSRL_pipeline_srl #:: resultVerbSRLPlusCoref_pipelineSRL #::
          srlV1ILP_path_lstm #:: resultSRLV1_curator #:: resultSimpleMatching #:: Stream.empty).find { t =>
          t._1._1.nonEmpty
        }
*/
        // no coref
//        (resultVerbSRLPlusCommaSRL_pathLSTM #:: resultSRLV3_curator #::
//          resultVerbSRLPlusCommaSRL_curatorSRL #:: resultVerbSRLPlusCommaSRL_pipelneSRL #:: resultSRLV3_pipeline #::
//          srlV1ILP_curator_srl #:: srlV1ILP_pipeline_srl #:: resultSRLV3_path_lstm #::
//          resultVerbSRLPlusPrepSRL_path_lstm #:: resultVerbSRLPlusPrepSRL_pipeline_srl #::
//          srlV1ILP_path_lstm #:: resultSRLV1_curator #:: resultSimpleMatching #:: Stream.empty).find { t =>
//          t._1._1.nonEmpty
//        }

         // no prep
//        (resultVerbSRLPlusCoref_curatorSRL #:: resultVerbSRLPlusCommaSRL_pathLSTM #:: resultSRLV3_curator #::
//          resultVerbSRLPlusCommaSRL_curatorSRL #:: resultVerbSRLPlusCommaSRL_pipelneSRL #:: resultSRLV3_pipeline #::
//          srlV1ILP_curator_srl #:: srlV1ILP_pipeline_srl #:: resultSRLV3_path_lstm #:: resultVerbSRLPlusCoref_pathLSTM #::
//          resultVerbSRLPlusCoref_pipelineSRL #::
//          srlV1ILP_path_lstm #:: resultSRLV1_curator #:: resultSimpleMatching #:: Stream.empty).find { t =>
//          t._1._1.nonEmpty
//        }
//
//        // no simple matching
        (resultVerbSRLPlusCoref_curatorSRL #:: resultVerbSRLPlusCommaSRL_pathLSTM #:: resultSRLV3_curator #::
          resultVerbSRLPlusCommaSRL_curatorSRL #:: resultVerbSRLPlusCommaSRL_pipelneSRL #:: resultSRLV3_pipeline #::
          srlV1ILP_curator_srl #:: srlV1ILP_pipeline_srl #:: resultSRLV3_path_lstm #:: resultVerbSRLPlusCoref_pathLSTM #::
          resultVerbSRLPlusPrepSRL_path_lstm #:: resultVerbSRLPlusPrepSRL_pipeline_srl #:: resultVerbSRLPlusCoref_pipelineSRL #::
          srlV1ILP_path_lstm #:: resultSRLV1_curator #:: Stream.empty).find { t =>
          t._1._1.nonEmpty
        }
      case TextILPModel.StackedForScienceMinimal =>
        // same mode as the previous one, with some curator components dropped
        (resultVerbSRLPlusCommaSRL_pathLSTM #:: resultVerbSRLPlusCommaSRL_pipelneSRL #:: resultSRLV3_pipeline #::
          srlV1ILP_pipeline_srl #:: resultSRLV3_path_lstm #:: resultVerbSRLPlusCoref_pathLSTM #::
          resultVerbSRLPlusPrepSRL_path_lstm #:: resultVerbSRLPlusPrepSRL_pipeline_srl #:: resultVerbSRLPlusCoref_pipelineSRL #::
          srlV1ILP_path_lstm #:: resultSimpleMatching #:: Stream.empty).find { t =>
          println("trying: " + t._2)
          t._1._1.nonEmpty
        }
      case TextILPModel.StackedForProcesses =>
        (resultWhatDoesItdo #:: resultCause #:: resultSRLV3_curator #:: resultSRLV3_path_lstm #::
          resultSRLV3_path_lstm #:: srlV1ILP_pipeline_srl #:: resultSRLV3_path_lstm #:: resultSRLV3_pipeline #::
          srlV1ILP_path_lstm #:: resultVerbSRLPlusPrepSRL_path_lstm #::
          srlV1ILP_curator_srl #:: srlV1ILP_path_lstm #:: resultVerbSRLPlusCoref_pathLSTM #::
          resultVerbSRLPlusPrepSRL_path_lstm #:: resultVerbSRLPlusCommaSRL_pathLSTM #:: resultSimpleMatching #:: Stream.empty).find { t =>
          println("trying: " + t._2)
          t._1._1.nonEmpty
        }
      case TextILPModel.StackedForProcessesMinimal =>
        (resultWhatDoesItdo #:: resultCause #:: resultSRLV3_path_lstm #::
          resultSRLV3_path_lstm #:: srlV1ILP_pipeline_srl #:: resultSRLV3_path_lstm #:: resultSRLV3_pipeline #::
          srlV1ILP_path_lstm #:: resultVerbSRLPlusPrepSRL_path_lstm #::
          srlV1ILP_path_lstm #:: resultVerbSRLPlusCoref_pathLSTM #::
          resultVerbSRLPlusPrepSRL_path_lstm #:: resultVerbSRLPlusCommaSRL_pathLSTM #:: resultSimpleMatching #:: Stream.empty).find { t =>
          println("trying: " + t._2)
          t._1._1.nonEmpty
        }
      case default =>
        // note: there is inefficiency here; we preprocess the input once at the beginning of this function, and also inside the linear classifier
        val result = EntityRelationResult()
        val selected = Seq(predictMaxScoreWekaClassifier(question, options, snippet))
        Some((selected, result) -> "Ensemble")
    }

    /*resultCause #:: resultWhatDoesItdo #:: resultVerbSRLPlusCoref_curatorSRL #:: resultVerbSRLPlusCoref_pathLSTM #::
        resultSRLV1_pipeline #:: resultVerbSRLPlusCoref_pipelineSRL #:: resultVerbSRLPlusPrepSRL_pipeline_srl #::
        srlV1ILP_pipeline_srl #:: resultVerbSRLPlusPrepSRL_path_lstm  #:: srlV1ILP_curator_srl #:: srlV1ILP_path_lstm #::*/
    /*resultVerbSRLPlusCommaSRL_pipelneSRL #:: resultVerbSRLPlusCommaSRL_pathLSTM #::*/
    /*resultVerbSRLPlusCommaSRL_curatorSRL #::*/
    /*resultSRLV3_pipeline*/
    /*#:: resultSimpleMatching*/

    //    val resultOpt = Seq(resultWhatDoesItdo, resultCause, resultSRLV1, resultVerbSRLPlusPrepSRL, srlV1ILP,
    //      resultVerbSRLPlusCoref, resultILP, resultSRLV2, resultVerbSRLPlusCommaSRL).find{ t =>
    //      println("trying: " + t._2)
    //      t._1._1.nonEmpty
    //    }

    //    val resultOpt = (
    //      resultWhatDoesItdo #::
    //        resultCause #::
    //        resultVerbSRLPlusCoref_pipelineSRL #::
    //        srlV1ILP_path_lstm #::
    //        resultVerbSRLPlusCoref_pathLSTM #::
    //        srlV1ILP_curator_srl #::
    //        resultSRLV1_pipeline #::
    //        resultVerbSRLPlusPrepSRL_path_lstm #::
    //        resultSimpleMatching #::
    //        resultVerbSRLPlusCoref_curatorSRL #::
    //        srlV1ILP_curator_srl #::
    //        srlV1ILP_pipeline_srl #::
    //        resultVerbSRLPlusPrepSRL_pipeline_srl #:: Stream.empty).find{ t =>
    //      println("trying: " + t._2)
    //      t._1._1.nonEmpty
    //    }

    /*
    List(
      WhatDoesItDoRule-SRL_VERB_PATH_LSTM.tsv,
      CauseRule-SRL_VERB_PATH_LSTM.tsv,
      VerbSRLandCoref-SRL_VERB.tsv,
      SRLV1ILP-SRL_VERB_PATH_LSTM.tsv,
      VerbSRLandCoref-SRL_VERB_PATH_LSTM.tsv,
      SRLV1ILP-SRL_VERB_CURATOR.tsv,
      SRLV1Rule-SRL_VERB.tsv,
      VerbSRLandPrepSRL-SRL_VERB_PATH_LSTM.tsv,
      SimpleMatching-SRL_VERB_PATH_LSTM.tsv,
      VerbSRLandCoref-SRL_VERB_CURATOR.tsv,
      SRLV1ILP-SRL_VERB.tsv,
      VerbSRLandPrepSRL-SRL_VERB.tsv,

      SRLV3Rule-SRL_VERB.tsv,
      VerbSRLandCommaSRL-SRL_VERB.tsv,
      VerbSRLandCommaSRL-SRL_VERB_CURATOR.tsv,
      VerbSRLandCommaSRL-SRL_VERB_PATH_LSTM.tsv,
      VerbSRLandPrepSRL-SRL_VERB_CURATOR.tsv,
      SRLV1Rule-SRL_VERB_PATH_LSTM.tsv,
      SRLV3Rule-SRL_VERB_PATH_LSTM.tsv,
      SRLV3Rule-SRL_VERB_CURATOR.tsv
    )
     */


    //    val resultOpt = (
    //        resultVerbSRLPlusCoref_pipelineSRL #:: srlV1ILP_pipeline_srl #::
    //          resultSimpleMatching #:: resultVerbSRLPlusPrepSRL_pipeline_srl #:: resultVerbSRLPlusCommaSRL_pipelneSRL #:: Stream.empty).find{ t =>
    //      println("trying: " + t._2)
    //      t._1._1.nonEmpty
    //    }
    //val resultOpt = Some(everything_jointly)

    //    val resultOpt = (resultVerbSRLPlusCoref_curatorSRL #:: srlV1ILP_pipeline_srl #:: resultVerbSRLPlusCoref_pipelineSRL #::
    //      srlV1ILP_curator_srl #:: resultVerbSRLPlusCoref_pathLSTM #:: resultVerbSRLPlusPrepSRL_pipeline_srl #:: srlV1ILP_path_lstm #::
    //      resultSRLV3_pipeline #:: resultVerbSRLPlusCommaSRL_pipelneSRL #:: resultVerbSRLPlusCommaSRL_curatorSRL #::
    //      resultVerbSRLPlusCommaSRL_pipelneSRL #:: Stream.empty).find{ t =>
    //      println("trying: " + t._2)
    //      t._1._1.nonEmpty
    //    }

    if (resultOpt.isDefined) {
      println(" ----> Selected method: " + resultOpt.get._2)
      resultOpt.get._1
    }
    else (Seq.empty, EntityRelationResult())
  }

  def solveWithReasoningType(question: String, options: Seq[String], snippet: String, reasoningType: ReasoningType, srlViewName: String = TextILPSolver.pathLSTMViewName): (Seq[Int], EntityRelationResult) = {
    val (q: Question, p: Paragraph) = preprocessQuestionData(question, options, snippet)
    reasoningType match {
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

  def solveWithHighestScore(question: String, options: Seq[String], snippet: String): (Seq[Int], EntityRelationResult) = {
    val (q: Question, p: Paragraph) = preprocessQuestionData(question, options, snippet)
    val types = Seq(SimpleMatching, VerbSRLandPrepSRL, SRLV1ILP, VerbSRLandCoref /*, VerbSRLandCommaSRL*/)
    val srlViewsAll = Seq(ViewNames.SRL_VERB /*, TextILPSolver.curatorSRLViewName, TextILPSolver.pathLSTMViewName*/)
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

  // what (does|do|can) .* do
  def WhatDoesItDoSolver(q: Question, p: Paragraph): (Seq[Int], EntityRelationResult) = {
    val qTokens = if (q.qTAOpt.get.hasView(ViewNames.SHALLOW_PARSE)) q.qTAOpt.get.getView(ViewNames.SHALLOW_PARSE).getConstituents.asScala else Seq.empty
    val pTokens = if (p.contextTAOpt.get.hasView(ViewNames.SHALLOW_PARSE)) p.contextTAOpt.get.getView(ViewNames.SHALLOW_PARSE).getConstituents.asScala else Seq.empty

    val selected = if (q.questionText.isWhatDoesItDo && q.answers.length == 2) {
      def getClosestIndex(qCons: Seq[Constituent], pCons: Seq[Constituent]): Int = {
        pCons.map { c =>
          TextILPSolver.getAvgScore(qCons, Seq(c))
        }.zipWithIndex.maxBy(_._1)._2
      }

      val qIdx = getClosestIndex(qTokens, pTokens)
      val ans1Cons = q.answers(0).aTAOpt.get.getView(ViewNames.TOKENS).asScala.toList
      val ans2Cons = q.answers(1).aTAOpt.get.getView(ViewNames.TOKENS).asScala.toList
      val a1Idx = getClosestIndex(ans1Cons, pTokens)
      val a2Idx = getClosestIndex(ans2Cons, pTokens)

      // one before, one after: after is the anser
      if (a1Idx < qIdx && a2Idx > qIdx) {
        // at least one of the answers happens before the question
        // the second one is the answer
        Seq(1)
      }
      else if (a1Idx > qIdx && a2Idx < qIdx) {
        // at least one of the answers happens before the question
        // the first one is the answer
        Seq(0)
      } else if (a1Idx > qIdx && a2Idx > qIdx) {
        // both after: closer is the answer
        if (a2Idx < a1Idx) {
          // at least one of the answers happens before the question
          // the second one is the answer
          Seq(1)
        }
        else if (a2Idx > a1Idx) {
          // the first one is the answer
          Seq(0)
        }
        else {
          Seq.empty
        }
      }
      else {
        Seq.empty
      }
    }
    else {
      Seq.empty
    }
    selected.distinct -> EntityRelationResult(log = "solved by WhatDoesItDoSolver", statistics = Stats(selected = true))
  }

  def SimilarityMetricSolver(q: Question, p: Paragraph): (Seq[Int], EntityRelationResult) = {
    def compareSimilairityInWndow(p: Paragraph, questionString: String, windowSize: Int, metric: StringMetric): Double = {
      val paragraphTokens = p.contextTAOpt.get.getTokens
      val paragraphSize = paragraphTokens.length
      if (paragraphSize > windowSize) {
        (0 until paragraphSize - windowSize).map { startIdx =>
          val endIdx = startIdx + windowSize
          val paragraphWindow = paragraphTokens.slice(startIdx, endIdx).mkString(" ")
          metric.compare(paragraphWindow, questionString)
        }.max
      } else {
        0.0
      }
    }

    // exact match: if there is a good match between the question
    val qparse = QuestionParse.constructFromString(q.questionText)
    val metric = StringMetrics.qGramsDistance()
    val fitbQuestionStrOpt = TextILPSolver.fitbGenerator.generateFITB(qparse).map(_.text)
    val selected = fitbQuestionStrOpt match {
      case Some(x) =>
        val indexScorePairs = q.answers.indices.map { idx =>
          val str = x.replace("BLANK_", q.answers(idx).answerText).dropRight(1).toLowerCase
          idx -> compareSimilairityInWndow(p, str, q.qTAOpt.get.getTokens.length +
            q.answers(idx).aTAOpt.get.getTokens.length, metric)
        }
        val sortedScores = indexScorePairs.sortBy(-_._2).take(2)
        val maxS = sortedScores(0)._2
        val secondMaxS = sortedScores(1)._2
        val maxIdx = sortedScores(0)._1
        if (maxS > params.exactMatchMinScoreValue && maxS - secondMaxS >= params.exactMatchMinScoreDiff) {
          // maxIdx should be the answer
          Seq(maxIdx)
        }
        else {
          Seq.empty
        }
      case None => Seq.empty
    }
    selected -> EntityRelationResult()
  }

  def CauseResultRules(q: Question, p: Paragraph): (Seq[Int], EntityRelationResult) = {
    // for result questions ...
    val answered = if (q.questionText.isForCResultQuestion && q.answers.length == 2) {
      def getClosestIndex(qCons: Seq[Constituent], pCons: Seq[Constituent]): Int = {
        pCons.map { c =>
          TextILPSolver.getAvgScore(qCons, Seq(c))
        }.zipWithIndex.maxBy(_._1)._2
      }

      val pCons = p.contextTAOpt.get.getView(ViewNames.SHALLOW_PARSE).asScala.toList
      val qCons = q.qTAOpt.get.getView(ViewNames.SHALLOW_PARSE).asScala.toList
      val qIdx = getClosestIndex(qCons, pCons)
      val ans1Cons = q.answers(0).aTAOpt.get.getView(ViewNames.TOKENS).asScala.toList
      val ans2Cons = q.answers(1).aTAOpt.get.getView(ViewNames.TOKENS).asScala.toList
      val a1Idx = getClosestIndex(ans1Cons, pCons)
      val a2Idx = getClosestIndex(ans2Cons, pCons)

      // at least one of the answers happens before the question
      if (a1Idx < qIdx && a2Idx > qIdx + 6) {
        // a2 should be the answer
        Seq(1)
      }
      else if (a2Idx < qIdx && a1Idx > qIdx + 6) {
        // a1 should be the answer
        Seq(0)
      }
      else {
        Seq.empty
      }
    }
    else {
      Seq.empty
    }
    answered -> EntityRelationResult(log = "solved by CauseResultRules", statistics = Stats(selected = true))
  }

  /**
    * select a frame from the paragraph such that:
    * Its A1/A0 argument has enough similarity with one of the terms in the questions
    * Its A1/A0 argument has enough similarity with the target answer option.
    */
  def SRLSolverV3(q: Question, p: Paragraph, alignmentFunction: AlignmentFunction, srlViewName: String): (Seq[Int], EntityRelationResult) = {
    val uniqueSelected = if (q.qTAOpt.get.hasView(srlViewName) && p.contextTAOpt.get.hasView(srlViewName)) {
      val qCons = q.qTAOpt.get.getView(ViewNames.SHALLOW_PARSE).getConstituents.asScala.map(_.getSurfaceForm)
      val pSRLView = p.contextTAOpt.get.getView(srlViewName)
      val pSRLCons = pSRLView.getConstituents.asScala
      val consGroupepd = pSRLCons.filter(c => c.getLabel != "Predicate" /*&& (c.getLabel == "A0" || c.getLabel == "A1")*/).flatMap { c =>
        c.getIncomingRelations.asScala.map(r => (c.getSurfaceForm, getPredicateFullLabel(r.getSource)))
      }.groupBy(_._2)

      val bothAnswersCovered = q.answers.zipWithIndex.forall { case (ans, idx) =>
        pSRLCons.exists { c => alignmentFunction.scoreCellQChoice(ans.answerText, c.getSurfaceForm) > 0.25 }
      }

      val framesThatAlignWithQuestion = consGroupepd.filter { case (pred, list) =>
        val cons = list.unzip._1
        qCons.zip(cons).foreach { case (qC, pC) =>
        }
        qCons.zip(cons).exists { case (qC, pC) => alignmentFunction.scoreCellQCons(qC, pC) > 0.15 }
      }

      val scorePerAns = q.answers.zipWithIndex.map { case (ans, idx) =>
        (framesThatAlignWithQuestion.flatMap { case (pred, list) =>
          val cons = list.unzip._1
          cons.map(c => alignmentFunction.scoreCellQChoice(ans.answerText, c))
        } ++ Seq(-1.0)).max
      }
      val selected = Seq(scorePerAns.zipWithIndex.maxBy(_._1)._2)
      if (scorePerAns.zipWithIndex.maxBy(_._1)._1 == -1 || bothAnswersCovered) Seq.empty else selected
    }
    else {
      Seq.empty
    }

    uniqueSelected -> EntityRelationResult(log = s"Solved by SRLV3 / View: $srlViewName  / selected: $uniqueSelected", statistics = Stats(selected = true))
  }

  def SRLSolverV2(q: Question, p: Paragraph, srlViewName: String): (Seq[Int], EntityRelationResult) = {
    assert(q.qTAOpt.get.hasView(annotationUtils.fillInBlankAnnotator.getViewName),
      "q.questionText: " + q.questionText + " / the current view: " + q.qTAOpt.get.getAvailableViews.asScala)
    val fillInBlank = q.qTAOpt.get.getView(annotationUtils.fillInBlankAnnotator.getViewName).getConstituents.get(0).getLabel
    val fillInBlankTA = if (fillInBlank == "") {
      q.qTAOpt.get
    } else {
      annotationUtils.pipelineServerClient.annotate(fillInBlank)
    }
    val results = q.answers.map { ans =>
      require(ans.aTAOpt.get.hasView(ViewNames.SHALLOW_PARSE), s"Answer TA does not contain shallow parse view. fillInBlank: $fillInBlank - Views are: ${ans.aTAOpt.get.getAvailableViews}")
      require(fillInBlankTA.hasView(ViewNames.SHALLOW_PARSE), s"fillInBlankTA does not contain shallow parse view. Views are: ${fillInBlankTA.getAvailableViews}")
      val questionAndAnsOption = annotationUtils.blankQuestionAnswerOptionNormalizer(ans.aTAOpt.get, fillInBlankTA, annotationUtils)
      val questionAndAnsOptionTA = annotationUtils.pipelineServerClient.annotate(questionAndAnsOption)
      if (questionAndAnsOptionTA.hasView(srlViewName) && p.contextTAOpt.get.hasView(srlViewName)) {
        if (matchPredicatesAndArguments(questionAndAnsOptionTA, p.contextTAOpt.get, ans, keytermsWithWHOverlap = false,
          doOverlapWithQuestion = true, fillInBlank, srlViewName)) 1.0 else 0.0
      }
      else {
        0.0
      }
    }
    if (verbose) println("results: " + results)
    val selectedAnswers = if (results.sum == 1) {
      // i.e. only one of the answer options has overlap
      results.zipWithIndex.flatMap { case (v, idx) =>
        if (v == 1) {
          Seq(idx)
        }
        else {
          Seq.empty
        }
      }
    }
    else {
      Seq.empty
    }

    selectedAnswers.distinct -> EntityRelationResult(log = s"Solved by SRLV2 / View: $srlViewName  / selected: $selectedAnswers", statistics = Stats(selected = true))
  }

  def SRLSolverV1(q: Question, p: Paragraph, srlViewName: String): (Seq[Int], EntityRelationResult) = {
    val selectedAnswers = if (q.qTAOpt.get.hasView(srlViewName) && p.contextTAOpt.get.hasView(srlViewName)) {
      val results = q.answers.map { ans =>
        if (matchPredicatesAndArguments(q.qTAOpt.get, p.contextTAOpt.get, ans, keytermsWithWHOverlap = true,
          doOverlapWithQuestion = false, q.questionText, srlViewName)) 1.0 else 0.0
      }
      if (verbose) println("results: " + results)
      if (results.sum == 1) {
        // i.e. only one of the answer options has overlap
        results.zipWithIndex.flatMap { case (v, idx) =>
          if (v == 1) {
            Seq(idx)
          }
          else {
            Seq.empty
          }
        }
      }
      else {
        Seq.empty
      }
    }
    else {
      Seq.empty
    }

    selectedAnswers.distinct -> EntityRelationResult(log = s"Solved by SRLV1 / View: $srlViewName  / selected: $selectedAnswers", statistics = Stats(selected = true))
  }

  // input supposed to be a predicate (i.e. it gets connected to other arguments via outgoing edges)
  def getConstituentsInFrame(c: Constituent): Seq[Constituent] = {
    c.getOutgoingRelations.asScala.map(_.getTarget)
  }

  def getTokensInFrame(c: Constituent): Seq[String] = {
    getConstituentsInFrame(c).flatMap(_.getSurfaceForm.split(" "))
  }

  def matchPredicatesAndArguments(qTA: TextAnnotation,
                                  pTA: TextAnnotation, ans: Answer,
                                  keytermsWithWHOverlap: Boolean = true,
                                  doOverlapWithQuestion: Boolean = false,
                                  originalQuestion: String, srlViewName: String): Boolean = {
    val verbose = false
    val qSRLView = qTA.getView(srlViewName)
    val pSRLView = pTA.getView(srlViewName)
    val qSRLCons = qSRLView.getConstituents.asScala
    val pSRLCons = pSRLView.getConstituents.asScala

    val keytermCons = if (keytermsWithWHOverlap) {
      // does question-srl contain any question term?
      qSRLCons.filter(c => SolverUtils.questionTerms.contains(c.getSurfaceForm.toLowerCase) && c.getIncomingRelations.size() > 0)
    }
    else {
      // align with answer words
      if (qTA.text.contains(ans.answerText.trim)) {
        val idxBegin = qTA.text.indexOf(ans.answerText)
        val idxEnd = idxBegin + ans.answerText.length - 1
        val out = qSRLCons.filter { c =>
          (c.getEndCharOffset <= idxEnd && c.getEndCharOffset >= idxBegin) ||
            (c.getStartCharOffset <= idxEnd && c.getStartCharOffset >= idxBegin) ||
            (c.getEndCharOffset >= idxEnd && c.getStartCharOffset <= idxBegin)
        }.filter(c => c.getLabel != "Predicate" && (c.getLabel == "A1" || c.getLabel == "A1") && c.getIncomingRelations.size() > 0)
        if (out.isEmpty) {
          // println("qSRLCons: " + qSRLCons)
        }
        out
      }
      else {
        Seq.empty
      }
    }

    val questionWithoutKeyTermsTokens = originalQuestion.split(" ").toSet.diff(SolverUtils.stopwords)

    keytermCons.exists { cons =>
      val qArgLabel = cons.getLabel
      val qSource = cons.getIncomingRelations.get(0).getSource

      val qSourceLabel = getPredicateFullLabel(qSource)
      val pParagraph = pSRLCons.filter(c => getPredicateFullLabel(c) == qSourceLabel)
      val pArgCons = pParagraph.flatMap { pred =>
        val overlapWithQuestion = questionWithoutKeyTermsTokens.intersect(getTokensInFrame(pred).toSet)
        val answerOptionContainsThePredicate = ans.answerText.contains(qSource.getSurfaceForm.trim)
        val containsOneOfQuestionTerms = if (answerOptionContainsThePredicate) overlapWithQuestion.nonEmpty else true
        if (containsOneOfQuestionTerms) {
          pred.getOutgoingRelations.asScala.map(_.getTarget).filter(_.getLabel == qArgLabel)
        }
        else {
          Seq.empty
        }
      }

      // check if any of the any of t he selected constituents overlap with one of the answers
      def getClosestIndex(qCons: Seq[Constituent], pCons: Seq[Constituent]): Int = {
        pCons.map { c =>
          TextILPSolver.getAvgScore(qCons, Seq(c))
        }.zipWithIndex.maxBy(_._1)._2
      }

      assert(pTA.hasView(ViewNames.SHALLOW_PARSE), "current views: " + pTA.getAvailableViews.asScala)
      val pCons = pTA.getView(ViewNames.SHALLOW_PARSE).asScala.toList
      val ansCons = ans.aTAOpt.get.getView(ViewNames.TOKENS).asScala.toList
      val ansParagraphIdx = getClosestIndex(ansCons, pCons)
      val ansSpan = pCons(ansParagraphIdx).getSpan
      if (pArgCons.exists { pArg => pArg.getSpan.getFirst <= ansSpan.getFirst && pArg.getSpan.getSecond >= ansSpan.getSecond }) true else false
    }
  }

  def getPredicateFullLabel(pred: Constituent): String = pred.getLabel + pred.getAttribute("SenseNumber") + pred.getAttribute("predicate")

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

      // weight for answers being close to each other
      /*val weight = -1.0
      aTokens.indices.map{ answerOptionIdx =>
        // for each answer option create one of these weights
        //val x = ilpSolver.createIntegerVar("answerTokenDistanceWeights", 0, 100, 1.0)
        aTokens.foreach{ toks =>
          val x = ilpSolver.createIntegerVar("answerTokenDistanceWeights", 0, 100, 1.0)
        }
        val len = aTokens(answerOptionIdx).length - 1
        for{
          i <- 0 until len - 1
          j <- i + 1 until len
        } {
          val x = ilpSolver.createIntegerVar(s"Answer:$answerOptionIdx-iAndjAnswerOption$i-$j", 0, 100, weight)
          // add constraint

        }

        (answerOptionIdx, x)
      }*/

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

    if (reasoningTypes.contains(SimpleMatchingWithCoref)) {
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

      // weight for answers being close to each other
      /*val weight = -1.0
      aTokens.indices.map{ answerOptionIdx =>
        // for each answer option create one of these weights
        //val x = ilpSolver.createIntegerVar("answerTokenDistanceWeights", 0, 100, 1.0)
        aTokens.foreach{ toks =>
          val x = ilpSolver.createIntegerVar("answerTokenDistanceWeights", 0, 100, 1.0)
        }
        val len = aTokens(answerOptionIdx).length - 1
        for{
          i <- 0 until len - 1
          j <- i + 1 until len
        } {
          val x = ilpSolver.createIntegerVar(s"Answer:$answerOptionIdx-iAndjAnswerOption$i-$j", 0, 100, weight)
          // add constraint

        }

        (answerOptionIdx, x)
      }*/

      // extra weight for alignment of paragraph constituents
      // create edges between constituents which have an edge in the dependency parse
      // this edge can be active only if the base nodes are active
      def twoAnswerConsAreConnectedViaDependencyParse(ansIdx: Int, tokIdx1: Int, tokIdx2: Int): Boolean = {
        val cons1 = getAnswerOptionCons(ansIdx, tokIdx1)
        val cons2 = getAnswerOptionCons(ansIdx, tokIdx2)
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

    // in this reasoning, we match the frames of the question with the frames of the paragraphs directly
    if (reasoningTypes.contains(SRLV1ILP)) {
      val qVerbConstituents = if (qTA.hasView(srlViewName)) qTA.getView(srlViewName).getConstituents.asScala else Seq.empty
      val pVerbConstituents = if (pTA.hasView(srlViewName)) pTA.getView(srlViewName).getConstituents.asScala else Seq.empty
      val qVerbPredicates = qVerbConstituents.filter(_.getLabel == "Predicate")
      val pVerbPredicates = pVerbConstituents.filter(_.getLabel == "Predicate")
      val qVerbArguments = qVerbConstituents diff qVerbPredicates
      val pVerbArguments = pVerbConstituents diff pVerbPredicates

      val pVerbPredicateToArgumentMap = pVerbConstituents.map(c => c -> c.getOutgoingRelations.asScala.map(_.getTarget)).toMap
      val qVerbPredicateToArgumentMap = qVerbConstituents.map(c => c -> c.getOutgoingRelations.asScala.map(_.getTarget)).toMap

      // active question verb-srl constituents
      val activeQuestionVerbSRLConstituents = qVerbConstituents.zipWithIndex.map { case (c, idx) =>
        // TODO: tune this weight
        c -> ilpSolver.createBinaryVar(s"activeQuestionVerbCons=$idx", 0.001)
      }.toMap

      // active paragraph verb-srl constituents
      val activeParagraphVerbSRLConstituents = pVerbConstituents.zipWithIndex.map { case (c, idx) =>
        // TODO: tune this weight
        c -> ilpSolver.createBinaryVar(s"activeParagraphVerbCons=$idx", 0.001)
      }.toMap

      // constraint: have some constituents used from the question
      // ilpSolver.addConsAtLeastK(s"", activeQuestionVerbSRLConstituents.values.toSeq, 2.0)

      // constraint: a predicate from questions should be used
      ilpSolver.addConsAtLeastK(s"", qVerbPredicates.map(activeQuestionVerbSRLConstituents), 1.0)

      // constraint: a argument from questions should be used
      ilpSolver.addConsAtLeastK(s"", qVerbArguments.map(activeQuestionVerbSRLConstituents), 1.0)

      // constraint:
      // have at most k active srl-verb predicates in the paragraph
      val predicateVariables = pVerbPredicates.map(activeParagraphVerbSRLConstituents)
      ilpSolver.addConsAtMostK(s"", predicateVariables, 1.0) // TODO: tune this number

      // QP alignments: alignment between question verb-srl argument and paragraph verb-srll constituents
      val argumentAlignments = {
        for {
          qVerbC <- qVerbArguments
          pVerbC <- pVerbArguments
          score = alignmentFunction.scoreCellQCons(qVerbC.getSurfaceForm, pVerbC.getSurfaceForm)
          if score >= 0.60 // TODO: tune this
        } yield {
          val x = ilpSolver.createBinaryVar("", score)

          // constraint: if pairwise variable is active, the two end should be active too.
          val qVerbCVar = activeQuestionVerbSRLConstituents(qVerbC)
          val pVerbCVar = activeParagraphVerbSRLConstituents(pVerbC)
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint1", Seq(x, qVerbCVar), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint2", Seq(x, pVerbCVar), Seq(1.0, -1.0), None, Some(0.0))
          (qVerbC, pVerbC, x)
        }
      }

      // QP alignments: alignment between question verb-srl predicates and paragraph verb-srl predicates
      val predicateAlignments = {
        for {
          qVerbC <- qVerbPredicates
          pVerbC <- pVerbPredicates
          score = alignmentFunction.scoreCellQCons(qVerbC.getSurfaceForm, pVerbC.getSurfaceForm)
          if score >= 0.60 // TODO: tune this
        } yield {
          val x = ilpSolver.createBinaryVar("", score)

          // constraint: if pairwise variable is active, the two end should be active too.
          val qVerbCVar = activeQuestionVerbSRLConstituents(qVerbC)
          val pVerbCVar = activeParagraphVerbSRLConstituents(pVerbC)
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint1", Seq(x, qVerbCVar), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint2", Seq(x, pVerbCVar), Seq(1.0, -1.0), None, Some(0.0))
          (qVerbC, pVerbC, x)
        }
      }

      def getEdgesConnectedToQuestionCons(c: Constituent): Seq[V] = {
        predicateAlignments.filter(_._1 == c).map(_._3) ++ argumentAlignments.filter(_._1 == c).map(_._3)
      }

      // constrain: if any of the question constituents were active, at least one of their edges should be active too
      qVerbConstituents.foreach { qVerbC =>
        val qVerbCVar = activeQuestionVerbSRLConstituents(qVerbC)
        val connected = getEdgesConnectedToQuestionCons(qVerbC)
        val weights = 1.0 +: connected.map(_ => -1.0)
        ilpSolver.addConsBasicLinear(s"", qVerbCVar +: connected, weights, None, Some(0.0))
      }

      // PA alignments: alignment between paragraph verb-srl arguments and answer options
      val argumentAnswerAlignments = {
        for {
          pVerbC <- pVerbArguments
          (ansIdx, ansVar) <- activeAnswerOptions
          score = alignmentFunction.scoreCellCell(pVerbC.getSurfaceForm, q.answers(ansIdx).answerText)
          if score >= 0.65 // TODO: tune this
        } yield {
          val x = ilpSolver.createBinaryVar("", score)

          // constraint: if pairwise variable is active, the two end should be active too.
          val verbC = activeParagraphVerbSRLConstituents(pVerbC)
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint1", Seq(x, verbC), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint2", Seq(x, verbC), Seq(1.0, -1.0), None, Some(0.0))
          (pVerbC, ansIdx, 0, x)
        }
      }

      // constraint: predicate should be inactive, unless it has at least two incoming outgoing relations
      // pred * 2 <= \sum arguments
      val verbSRLEdges = pVerbPredicates.flatMap { pred =>
        val arguments = pVerbPredicateToArgumentMap(pred)
        val predVar = activeParagraphVerbSRLConstituents(pred)
        val argumentVars = arguments.map(activeParagraphVerbSRLConstituents)
        val weights = argumentVars.map(_ => -1.0)
        //ilpSolver.addConsBasicLinear(s"", predVar +: argumentVars, 2.0 +: weights, None, Some(0.0))

        // if predicate is inactive, nothing can be active
        argumentVars.foreach { v =>
          ilpSolver.addConsBasicLinear(s"", Seq(v, predVar), Seq(1.0, -1.0), None, Some(0.0))
        }
        //ilpSolver.addConsBasicLinear(s"", predVar +: argumentVars, 10.0 +: weights, Some(0.0), None)

        // get variables for SRL edges
        arguments.map { arg =>
          val argumentVar = activeParagraphVerbSRLConstituents(arg)
          // if both edges are active, visualize an edge
          val x = ilpSolver.createBinaryVar("", 0.01)
          ilpSolver.addConsBasicLinear(s"", Seq(x, predVar), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"", Seq(x, argumentVar), Seq(1.0, -1.0), None, Some(0.0))
          (pred, arg, x)
        }
      }

      questionParagraphAlignments ++= argumentAlignments.toBuffer ++ predicateAlignments.toBuffer
      interParagraphAlignments ++= verbSRLEdges.toBuffer

      // constraint: the predicate can be active only if at least one of the its connected arguments are active
      pVerbPredicates.foreach { predicate =>
        val predicateVar = activeParagraphVerbSRLConstituents(predicate)
        val arguments = pVerbPredicateToArgumentMap(predicate)
        val argumentsVars = arguments.map(activeParagraphVerbSRLConstituents)
        val weights = arguments.map(_ => 1.0)
        ilpSolver.addConsBasicLinear(s"", argumentsVars :+ predicateVar, weights :+ -1.0, Some(0.0), None)
      }

      paragraphAnswerAlignments ++= argumentAnswerAlignments.toBuffer
    }

    if (reasoningTypes.contains(VerbSRLandCommaSRL)) {
      val commaSRLPredicateabels = Set("Attribute", "Complementary", "Interrupter", "Introductory", "List", "Quotation", "Substitute", "Locative")

      val qVerbConstituents = if (qTA.hasView(srlViewName)) qTA.getView(srlViewName).getConstituents.asScala else Seq.empty
      val pVerbConstituents = if (pTA.hasView(srlViewName)) pTA.getView(srlViewName).getConstituents.asScala else Seq.empty
      //val qCommaConstituents = if (qTA.hasView(ViewNames.SRL_COMMA)) qTA.getView(ViewNames.SRL_COMMA).getConstituents.asScala else Seq.empty
      val pCommaConstituents = if (pTA.hasView(ViewNames.SRL_COMMA)) pTA.getView(ViewNames.SRL_COMMA).getConstituents.asScala else Seq.empty
      val pCommaPredicates = pCommaConstituents.filter(c => commaSRLPredicateabels.contains(c.getLabel))
      val qVerbPredicates = qVerbConstituents.filter(_.getLabel == "Predicate")
      val pVerbPredicates = pVerbConstituents.filter(_.getLabel == "Predicate")
      val qVerbArguments = qVerbConstituents diff qVerbPredicates
      val pVerbArguments = pVerbConstituents diff pVerbPredicates
      val pCommaArguments = pCommaConstituents diff pCommaPredicates
      val pCommaPredicateToArgumentMap = pCommaConstituents.map(c => c -> c.getOutgoingRelations.asScala.map(_.getTarget)).toMap
      val pVerbPredicateToArgumentMap = pVerbConstituents.map(c => c -> c.getOutgoingRelations.asScala.map(_.getTarget)).toMap
      val qVerbPredicateToArgumentMap = qVerbConstituents.map(c => c -> c.getOutgoingRelations.asScala.map(_.getTarget)).toMap


      // constraint:
      // (1) if any of the constituents in the frame are active, frame should be active
      // (2) if the frame is active, at least predicate and and one argument should be active
      def addConsistencyMapToFrames(predicateToArgumentsMap: Map[Constituent, Seq[Constituent]],
                                    constituents: Map[Constituent, V],
                                    frames: Seq[(Constituent, V)]) = {
        frames.foreach { case (pred, frameVariable) =>
          val argumentVariables = predicateToArgumentsMap(pred).map(constituents)
          val predicateVariable = constituents(pred)
          val allVariables = argumentVariables :+ predicateVariable
          val allWeights = allVariables.map(_ => 1.0)
          //if(activeConstaints) {
          // if the frame is active, at least one of its elements should be active
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint", allVariables :+ frameVariable, allWeights :+ -1.0, Some(0.0), None)
          allVariables.foreach { x =>
            // if any of the variables in the frame are active, the frame should be active
            ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, frameVariable), Seq(1.0, -1.0), None, Some(0.0))
          }
          //}
        }
      }

      // active paragraph comma-srl constituents
      val activeParagrapCommaSRLConstituents = pCommaConstituents.zipWithIndex.map { case (c, idx) =>
        // TODO: tune this weight
        c -> ilpSolver.createBinaryVar(s"activeParagraphCommaCons=$idx", 0.01)
      }.toMap

      // active question verb-srl frames
      // for each predicate create a variable
      //      val activeParagraphCommaSrlFrames = pCommaPredicates.map{ c =>
      //        // TODO: tune this weight
      //        c -> ilpSolver.createBinaryVar(s"activeQuestionFrame=$c", 0.1)
      //      }
      //
      //      println("activeParagrapCommaSRLConstituents: " + activeParagrapCommaSRLConstituents.size)
      //      println("activeParagraphCommaSrlFrames: " + activeParagraphCommaSrlFrames.size)
      //
      //      addConsistencyMapToFrames(pCommaPredicateToArgumentMap, activeParagrapCommaSRLConstituents, activeParagraphCommaSrlFrames)

      // active question verb-srl constituents
      val activeQuestionVerbSRLConstituents = qVerbConstituents.zipWithIndex.map { case (c, idx) =>
        // TODO: tune this weight
        c -> ilpSolver.createBinaryVar(s"activeQuestionVerbCons=$idx", 0.01)
      }.toMap
      // active question verb-srl frames
      // for each predicate create a variable
      //      val activeQuestionVerbSrlFrames = qVerbPredicates.map{ c =>
      //        // TODO: tune this weight
      //        c -> ilpSolver.createBinaryVar(s"activeQuestionFrame=$c", 0.1)
      //      }
      //
      //      println("activeQuestionVerbSRLConstituents: " + activeQuestionVerbSRLConstituents.size)
      //      println("activeQuestionVerbSrlFrames: " + activeQuestionVerbSrlFrames.size)
      //
      //      addConsistencyMapToFrames(qVerbPredicateToArgumentMap, activeQuestionVerbSRLConstituents, activeQuestionVerbSrlFrames)

      // active paragraph verb-srl constituents
      val activeParagraphVerbSRLConstituents = pVerbConstituents.zipWithIndex.map { case (c, idx) =>
        // TODO: tune this weight
        c -> ilpSolver.createBinaryVar(s"activeParagraphVerbCons=$idx", 0.01)
      }.toMap
      // active paragraph verb-srl frames
      // for each predicate create a variable
      //      val activeParagraphVerbSrlFrames = pVerbPredicates.map{ c =>
      //        // TODO: tune this weight
      //        c -> ilpSolver.createBinaryVar(s"activeQuestionFrame=$c", 0.1)
      //      }
      //
      //      println("activeParagraphVerbSRLConstituents: " + activeParagraphVerbSRLConstituents.size)
      //      println("activeParagraphVerbSrlFrames: " + activeParagraphVerbSrlFrames.size)
      //
      //      addConsistencyMapToFrames(pVerbPredicateToArgumentMap, activeParagraphVerbSRLConstituents, activeParagraphVerbSrlFrames)

      // constraint:
      // have some constituents used from the question
      ilpSolver.addConsAtLeastK(s"", activeQuestionVerbSRLConstituents.values.toSeq, 1.0)

      // PP alignments: alignment between paragraph comma-srl argument and paragraph verb-srl argument
      val pCommaPVerbAlignments = for {
        verbC <- pVerbArguments
        commaC <- pCommaArguments
        score = TextILPSolver.sahandClient.getScore(verbC.getSurfaceForm, commaC.getSurfaceForm, SimilarityNames.phrasesim)
        if score >= 0.75 // TODO: tune this
      }
        yield {
          //println("\t\t\tPP alignments: verbC: " + verbC.getSurfaceForm + " / commaC: " + commaC.getSurfaceForm + " / score: " + score)
          val x = ilpSolver.createBinaryVar("", score)
          // constraint: if pairwise variable is active, the two end should be active too.
          val srlVerbVar = activeParagraphVerbSRLConstituents(verbC)
          val srlCommaVar = activeParagrapCommaSRLConstituents(commaC)
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, srlVerbVar), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, srlCommaVar), Seq(1.0, -1.0), None, Some(0.0))
          (verbC, commaC, x)
        }

      // PP: inter-comma alignments
      // there is alignmet between arguments of the comma, if both of them are aligned
      val pCommaArgumentAglinments = for {
        pCommaPredicate <- pCommaPredicates
        if pCommaPredicate.getLabel == "introductory"
        arguments = pCommaPredicateToArgumentMap(pCommaPredicate)
        if arguments.length == 2
      }
        yield {
          assert(arguments.length == 2, s"the number of arguments connected to comma is ${arguments.length}")
          val arg1 = arguments(0)
          val arg2 = arguments(1)
          val x = ilpSolver.createBinaryVar("", 0.02) // TODO: tune this

          // constraint: if pairwise variable is active, the two end should be active too.
          val arg1Var = activeParagrapCommaSRLConstituents(arg1)
          val arg2Var = activeParagrapCommaSRLConstituents(arg2)
          //if(activeConstaints) {
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, arg1Var), Seq(1.0, -1.0), Some(0.0), Some(0.0))
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, arg2Var), Seq(1.0, -1.0), Some(0.0), Some(0.0))
          //}
          (arg1, arg2, x)
        }

      // QP alignments: alignment between question ver-srl argument paragraph comma-srl argument
      val pCommaQVerbAlignments = {
        for {
          verbC <- qVerbArguments
          commaC <- pCommaArguments
          score = alignmentFunction.scoreCellQCons(verbC.getSurfaceForm, commaC.getSurfaceForm)
          if score >= 0.50 // TODO: tune this
        } yield {
          val x = ilpSolver.createBinaryVar("", score)

          // constraint: if pairwise variable is active, the two end should be active too.
          val srlVerbVar = activeQuestionVerbSRLConstituents(verbC)
          val srlCommaVar = activeParagrapCommaSRLConstituents(commaC)
          //if (activeConstaints) {
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint1", Seq(x, srlVerbVar), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint2", Seq(x, srlCommaVar), Seq(1.0, -1.0), None, Some(0.0))
          //}
          (verbC, commaC, x)
        }
      }

      // QP alignments: predicate-predicate
      val pVerbQVerbAlignments = for {
        qVerbC <- qVerbPredicates
        pVerbC <- pVerbPredicates
        score = TextILPSolver.sahandClient.getScore(pVerbC.getSurfaceForm,
          qVerbC.getSurfaceForm, SimilarityNames.phrasesim) //alignmentFunction.scoreCellCell(pVerbC.getSurfaceForm, qVerbC.getSurfaceForm)
        if score >= 0.15 // TODO: tune this
      } yield {
        val x = ilpSolver.createBinaryVar("", score)
        //println(s"verb alignments: (${pVerbC.getSurfaceForm}, ${qVerbC.getSurfaceForm}) = ${score}")
        // constraint: if pairwise variable is active, the two end should be active too.
        val qVerbVar = activeQuestionVerbSRLConstituents(qVerbC)
        val pVerbVar = activeParagraphVerbSRLConstituents(pVerbC)
        ilpSolver.addConsBasicLinear(s"", Seq(x, qVerbVar), Seq(1.0, -1.0), None, Some(0.0))
        ilpSolver.addConsBasicLinear(s"", Seq(x, pVerbVar), Seq(1.0, -1.0), None, Some(0.0))
        (qVerbC, pVerbC, x)
      }

      // constraint:
      // question cons can be active, only if any least one of the edges connected to it is active
      def getEdgesConnectedToQuestionCons(c: Constituent): Seq[V] = {
        pVerbQVerbAlignments.filter(_._1 == c).map(_._3) ++ pCommaQVerbAlignments.filter(_._1 == c).map(_._3)
      }

      activeQuestionVerbSRLConstituents.foreach { case (c, x) =>
        val edges = getEdgesConnectedToQuestionCons(c)
        val weights = edges.map(_ => -1.0)
        ilpSolver.addConsBasicLinear(s"", x +: edges, 1.0 +: weights, None, Some(0.0))
      }

      // constraint: predicate should be inactive, unless it has at least two incoming outgoing relations
      // pred * 2 <= \sum arguments
      val verbSRLEdges = pVerbPredicates.flatMap { pred =>
        val arguments = pVerbPredicateToArgumentMap(pred)
        val predVar = activeParagraphVerbSRLConstituents(pred)
        val argumentVars = arguments.map(activeParagraphVerbSRLConstituents)
        val weights = argumentVars.map(_ => -1.0)
        //ilpSolver.addConsBasicLinear(s"", predVar +: argumentVars, 2.0 +: weights, None, Some(0.0))

        // if predicate is inactive, nothing can be active
        //ilpSolver.addConsBasicLinear(s"", predVar +: argumentVars, 10.0 +: weights, Some(0.0), None)


        // get variables for SRL edgess
        arguments.map { arg =>
          val argumentVar = activeParagraphVerbSRLConstituents(arg)
          // if both edges are active, visualize an edge
          val x = ilpSolver.createBinaryVar("", 0.01)
          ilpSolver.addConsBasicLinear(s"", Seq(x, predVar), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"", Seq(x, argumentVar), Seq(1.0, -1.0), None, Some(0.0))
          (pred, arg, x)
        }
      }

      questionParagraphAlignments ++= pCommaQVerbAlignments.toBuffer ++ pVerbQVerbAlignments.toBuffer
      interParagraphAlignments ++= pCommaPVerbAlignments.toBuffer ++ pCommaArgumentAglinments.toBuffer ++ verbSRLEdges.toBuffer

      // constraint: the predicate can be active only if at least one of the its connected arguments are active
      if (true) {
        pVerbPredicates.foreach { predicate =>
          val predicateVar = activeParagraphVerbSRLConstituents(predicate)
          val arguments = pVerbPredicateToArgumentMap(predicate)
          val argumentsVars = arguments.map(activeParagraphVerbSRLConstituents)
          val weights = arguments.map(_ => 1.0)
          ilpSolver.addConsBasicLinear(s"", argumentsVars :+ predicateVar, weights :+ -1.0, Some(0.0), None)
        }
      }

      // PA alignments: alignment between verb-srl argument in paragraph and answer option
      val ansPVerbAlignments = for {
        verbC <- pVerbArguments
        (ansIdx, ansVar) <- activeAnswerOptions
        score = alignmentFunction.scoreCellCell(verbC.getSurfaceForm, q.answers(ansIdx).answerText)
        if score >= 0.65 // TODO: tune this
      } yield {
        val x = ilpSolver.createBinaryVar("", score)

        // constraint: if pairwise variable is active, the two end should be active too.
        val srlVerbVar = activeParagraphVerbSRLConstituents(verbC)
        //if(activeConstaints) {
        ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, srlVerbVar), Seq(1.0, -1.0), None, Some(0.0))
        ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, ansVar), Seq(1.0, -1.0), None, Some(0.0))
        //}

        // constraint: if the argument is active, the predicate in the same frame should be active too.
        //        val predicate = verbC.getIncomingRelations.get(0).getSource
        //        val predicateVar = activeParagraphVerbSRLConstituents(predicate)
        //        ilpSolver.addConsBasicLinear(s"", Seq(srlVerbVar, predicateVar), Seq(1.0, -1.0), None, Some(0.0))

        // constraint: if the argument is active, one other argument in the same frame should be active too.
        //        val constituents = pVerbPredicateToArgumentMap(predicate).filter(_ != verbC)
        //        val constituentVars = constituents.map(activeParagraphVerbSRLConstituents)
        //        val weights = constituentVars.map(_ => -1.0)
        //        ilpSolver.addConsBasicLinear(s"", srlVerbVar +: constituentVars, 1.0 +: weights, None, Some(0.0))
        (verbC, ansIdx, 0, x)
      }

      // constraint: not more than one argument of a frame, should be aligned to answer option:
      ansPVerbAlignments.groupBy(_._1.getIncomingRelations.get(0).getSource).foreach { case (_, list) =>
        val variables = list.map(_._4)
        val weights = variables.map(_ => 1.0)
        ilpSolver.addConsBasicLinear(s"", variables, weights, None, Some(1.0))
      }

      paragraphAnswerAlignments ++= ansPVerbAlignments.toBuffer

      def getConstituentsConectedToParagraphSRLArg(c: Constituent): Seq[V] = {
        pCommaPVerbAlignments.filter(_._1 == c).map {
          _._3
        } ++ ansPVerbAlignments.filter(_._1 == c).map {
          _._4
        }
      }

      // constraint: no dangling arguments: i.e. any verb-srl argument in the paragram, should be connected to at least
      // one other thing, in addition to its predicate
      if (true) {
        pVerbArguments.zipWithIndex.foreach { case (arg, idx) =>
          val argVar = activeParagraphVerbSRLConstituents(arg)
          val connected = getConstituentsConectedToParagraphSRLArg(arg)
          if (connected.nonEmpty) {
            val weights = connected.map(_ => -1.0)
            ilpSolver.addConsBasicLinear(s"", argVar +: connected, 1.0 +: weights, None, Some(0.0))
          }
          else {
            // if nothing is connected to it, then it should be inactive
            ilpSolver.addConsBasicLinear(s"", Seq(argVar), Seq(1.0), None, Some(0.0))
          }
        }
      }

      def getConnectedEdgesToParagraphVerbPredicate(pred: Constituent): Seq[V] = {
        val pVerbQVerbAlignments1 = pVerbQVerbAlignments.filter(_._2 == pred)
        //val verbSRLEdges1 = verbSRLEdges.filter(_._1 == pred)
        /*verbSRLEdges1.map(_._3) ++*/ pVerbQVerbAlignments1.map(_._3)
      }

      pVerbPredicates.foreach { p =>
        // constraint:
        // for each predicate in the paragraph, it should not be active, unless it has at least one verb connected to it from question
        // 2 * pVar  <= \sum edges
        val edges = getConnectedEdgesToParagraphVerbPredicate(p)
        val pVar = activeParagraphVerbSRLConstituents(p)
        val weights = edges.map(_ => -1.0)
        ilpSolver.addConsBasicLinear(s"", pVar +: edges, 1.0 +: weights, None, Some(0.0))
      }
    }

    if (reasoningTypes.contains(VerbSRLandPrepSRL)) {
      val qVerbConstituents = if (qTA.hasView(srlViewName)) qTA.getView(srlViewName).getConstituents.asScala else Seq.empty
      val pVerbConstituents = if (pTA.hasView(srlViewName)) pTA.getView(srlViewName).getConstituents.asScala else Seq.empty
      val pPrepConstituents = if (pTA.hasView(ViewNames.SRL_PREP)) pTA.getView(ViewNames.SRL_PREP).getConstituents.asScala else Seq.empty
      val pPrepPredicates = if (pTA.hasView(ViewNames.SRL_PREP)) pTA.getView(ViewNames.SRL_PREP).asInstanceOf[PredicateArgumentView].getPredicates.asScala else Seq.empty
      val qVerbPredicates = qVerbConstituents.filter(_.getLabel == "Predicate")
      val pVerbPredicates = pVerbConstituents.filter(_.getLabel == "Predicate")
      val qVerbArguments = qVerbConstituents diff qVerbPredicates
      val pVerbArguments = pVerbConstituents diff pVerbPredicates
      val pPrepArguments = pPrepConstituents diff pPrepPredicates

      val pPrepPredicateToArgumentMap = pPrepPredicates.map(c => c -> c.getOutgoingRelations.asScala.map(_.getTarget)).toMap
      val pVerbPredicateToArgumentMap = pVerbPredicates.map(c => c -> c.getOutgoingRelations.asScala.map(_.getTarget)).toMap
      val qVerbPredicateToArgumentMap = qVerbPredicates.map(c => c -> c.getOutgoingRelations.asScala.map(_.getTarget)).toMap

      // active paragraph prep-srl constituents
      val activeParagraphPrepSRLConstituents = pPrepConstituents.zipWithIndex.map { case (c, idx) =>
        c -> ilpSolver.createBinaryVar(s"", 0.01) //TODO: tune this weight
      }.toMap

      // active question verb-srl constituents
      val activeQuestionVerbSRLConstituents = qVerbConstituents.zipWithIndex.map { case (c, idx) =>
        c -> ilpSolver.createBinaryVar(s"activeQuestionVerbCons=$idx", 0.01) //TODO: tune this weight
      }.toMap

      // active verb-srl frames in the question
      val activeQuestionVerbSRLFrames = qVerbPredicates.zipWithIndex.map { case (c, idx) =>
        c -> ilpSolver.createBinaryVar(s"", 0.05) //TODO: tune this weight
      }.toMap

      // active paragraph verb-srl constituents
      val activeParagraphVerbSRLConstituents = pVerbConstituents.zipWithIndex.map { case (c, idx) =>
        c -> ilpSolver.createBinaryVar(s"activeParagraphVerbCons=$idx", 0.01) //TODO: tune this weight
      }.toMap


      // constraint: if any of the constituents in the frame are active, the frame should be active
      qVerbPredicates.foreach { c =>
        val args = qVerbPredicateToArgumentMap(c)
        val vars = (c +: args).map(activeQuestionVerbSRLConstituents)
        val weights = vars.map(_ => -1.0)
        val activeFrameVar = activeQuestionVerbSRLFrames(c)
        ilpSolver.addConsBasicLinear(s"", activeFrameVar +: vars, 1.0 +: weights, None, Some(0.0))
      }

      // constraint: use at most two active frames in the question
      ilpSolver.addConsAtMostK(s"", activeQuestionVerbSRLFrames.unzip._2.toList, 2.0)

      val qVerbArgumentVars = qVerbArguments.map(activeQuestionVerbSRLConstituents)
      // constraint: use at least 1 verb-srl argument of the question
      ilpSolver.addConsAtLeastK(s"", qVerbArgumentVars, 1.0)

      // constraint: use at most 3 verb-srl argument of the question
      ilpSolver.addConsAtMostK(s"", qVerbArgumentVars, 3.0)

      // constraint: use at most 1 verb-srl arguments of each frame
      qVerbPredicateToArgumentMap.foreach { case (_, args) =>
        ilpSolver.addConsAtMostK(s"", args.map(activeQuestionVerbSRLConstituents), 1.0)
      }

      // PP alignments: alignment between paragraph prep-srl argument and paragraph verb-srl argument
      val pPrepVerbAlignments = for {
        verbC <- pVerbArguments
        prepC <- pPrepArguments
        score = TextILPSolver.sahandClient.getScore(verbC.getSurfaceForm, prepC.getSurfaceForm, SimilarityNames.phrasesim)
        if score >= 0.7 // TODO: tune this
      }
        yield {
          val x = ilpSolver.createBinaryVar("", score)
          // constraint: if pairwise variable is active, the two end should be active too.
          val srlVerbVar = activeParagraphVerbSRLConstituents(verbC)
          val srlPrepVar = activeParagraphPrepSRLConstituents(prepC)
          ilpSolver.addConsBasicLinear(s"", Seq(x, srlVerbVar), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"", Seq(x, srlPrepVar), Seq(1.0, -1.0), None, Some(0.0))
          (verbC, prepC, x)
        }

      // constraint: in prep-srl frames, both arguments have to have other incoming edges.
      def edgesConnectedToPrepSRLArguments(c: Constituent): Seq[V] = {
        pPrepVerbAlignments.filter(_._2 == c).map(_._3)
      }

      pPrepArguments.foreach { c =>
        val srlPrepVar = activeParagraphPrepSRLConstituents(c)
        val vars = edgesConnectedToPrepSRLArguments(c)
        val weights = vars.map(_ => -1.0)
        // if all vars are inactive, prep-srl-var should be inactive
        ilpSolver.addConsBasicLinear(s"", vars :+ srlPrepVar, weights :+ 1.0, None, Some(0.0))
      }

      // constraint: no loop allowed: loop of a verb-srl arg connected to two prep-srl args of the same frame
      pPrepVerbAlignments.groupBy { case (verbC, prepC, _) =>
        (prepC.getIncomingRelations.asScala.head.getSource, verbC)
      }.foreach { case (_, seq) =>
        val (verbCCons, prepCons, vars) = seq.unzip3
        require(verbCCons.distinct.size == 1)
        ilpSolver.addConsBasicLinear(s"", vars, vars.map(_ => 1.0), None, Some(1.0))
      }

      // PP: inter-prep alignments
      // alignments from prep-srl-predicates to prep-srl-arguments
      val pPrepArgumentAlignments = for {
        pPrepPredicate <- pPrepPredicates
        pVar = activeParagraphPrepSRLConstituents(pPrepPredicate)
        arguments = pPrepPredicateToArgumentMap(pPrepPredicate)
        arg <- arguments
        argVar = activeParagraphPrepSRLConstituents(arg)
      }
        yield {
          val x = ilpSolver.createBinaryVar("", 0.02) //TODO: tune this
          // constraint: if pairwise variable is active, the two end should be active too.
          ilpSolver.addConsBasicLinear(s"", Seq(x, argVar), Seq(1.0, -1.0), Some(0.0), Some(0.0))
          ilpSolver.addConsBasicLinear(s"", Seq(x, pVar), Seq(1.0, -1.0), Some(0.0), Some(0.0))
          // constraint: predicate can be active only if at leats one of its argument is active
          (pPrepPredicate, arg, x)
        }

      // constraint: use exactly one prep-srl predicate
      val predicateVars = pPrepPredicates.map(activeParagraphPrepSRLConstituents)
      val weights = predicateVars.map(_ => 1.0)
      ilpSolver.addConsBasicLinear(s"", predicateVars, weights, Some(1.0), Some(1.0))

      // constraint: use exactly one verb-srl predicate
      val predicateVars2 = pVerbPredicates.map(activeParagraphVerbSRLConstituents)
      val weights2 = predicateVars2.map(_ => 1.0)
      ilpSolver.addConsBasicLinear(s"", predicateVars2, weights2, Some(1.0), Some(2.0))

      // constraint: predicate can be active only if at least one of its argument is active
      //      pPrepPredicates.foreach{ pPrepPredicate =>
      //        val pVar = activeParagraphPrepSRLConstituents(pPrepPredicate)
      //        val arguments = pPrepPredicateToArgumentMap(pPrepPredicate)
      //        val argVars = arguments.map(activeParagraphPrepSRLConstituents)
      //        val weights = argVars.map(_ => -1.0)
      //        ilpSolver.addConsBasicLinear(s"", pVar +: argVars, 1.0 +: weights, None, Some(0.0))
      //      }

      // QP alignments: alignment between question verb-srl argument paragraph verb-srl argument
      val pVerbQVerbAlignments = {
        for {
          qVerbC <- qVerbArguments
          pVerbC <- pVerbArguments
          score = alignmentFunction.scoreCellCell(qVerbC.getSurfaceForm, pVerbC.getSurfaceForm)
          if score >= 0.60 // TODO: tune this
        } yield {
          val x = ilpSolver.createBinaryVar("", score)

          // constraint: if pairwise variable is active, the two end should be active too.
          val qSrlVerbVar = activeQuestionVerbSRLConstituents(qVerbC)
          val pSrlVerbVar = activeParagraphVerbSRLConstituents(pVerbC)
          ilpSolver.addConsBasicLinear(s"", Seq(x, qSrlVerbVar), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"", Seq(x, pSrlVerbVar), Seq(1.0, -1.0), None, Some(0.0))
          (qVerbC, pVerbC, x)
        }
      }

      // each verb-srl argument in the question can have eat most 1 outgoing edges
      pVerbQVerbAlignments.groupBy(_._1).foreach { case (_, seq) =>
        val edges = seq.unzip3._3
        ilpSolver.addConsAtMostK("", edges, 1)
      }

      // QP alignments: predicate-predicate
      val pVerbQVerbPredicateAlignments = for {
        qVerbC <- qVerbPredicates
        if !TextILPSolver.toBeVerbs.contains(qVerbC.getSurfaceForm)
        pVerbC <- pVerbPredicates
        score = TextILPSolver.sahandClient.getScore(pVerbC.getSurfaceForm, qVerbC.getSurfaceForm, SimilarityNames.phrasesim)
        if score >= 0.50 // TODO: tune this
      } yield {
        val x = ilpSolver.createBinaryVar("", score)
        // constraint: if pairwise variable is active, the two end should be active too.
        val qVerbVar = activeQuestionVerbSRLConstituents(qVerbC)
        val pVerbVar = activeParagraphVerbSRLConstituents(pVerbC)
        ilpSolver.addConsBasicLinear(s"", Seq(x, qVerbVar), Seq(1.0, -1.0), None, Some(0.0))
        ilpSolver.addConsBasicLinear(s"", Seq(x, pVerbVar), Seq(1.0, -1.0), None, Some(0.0))
        (qVerbC, pVerbC, x)
      }

      // constraint: question constituents can be active only if any of at least one of their QP edges are active
      def getEdgesForQuestionCons(c: Constituent): Seq[V] = {
        pVerbQVerbPredicateAlignments.filter(_._1 == c).map(_._3) ++ pVerbQVerbAlignments.filter(_._1 == c).map(_._3)
      }

      activeQuestionVerbSRLConstituents.foreach { case (c, x) =>
        val edges = getEdgesForQuestionCons(c)
        val weights = edges.map(_ => -1.0)
        ilpSolver.addConsBasicLinear(s"", x +: edges, 1.0 +: weights, None, Some(0.0))
      }

      // constraint: predicate should be inactive, unless it has at least two incoming outgoing relations
      // pred * 2 <= \sum arguments
      val verbSRLEdges = pVerbPredicates.flatMap { pred =>
        val arguments = pVerbPredicateToArgumentMap(pred)
        val predVar = activeParagraphVerbSRLConstituents(pred)
        val argumentVars = arguments.map(activeParagraphVerbSRLConstituents)
        val weights = argumentVars.map(_ => -1.0)
        //ilpSolver.addConsBasicLinear(s"", predVar +: argumentVars, 2.0 +: weights, None, Some(0.0))

        // if predicate is inactive, nothing can be active
        //ilpSolver.addConsBasicLinear(s"", predVar +: argumentVars, 10.0 +: weights, Some(0.0), None)

        // get variables for SRL edges
        arguments.map { arg =>
          val argumentVar = activeParagraphVerbSRLConstituents(arg)
          // if both edges are active, visualize an edge
          val x = ilpSolver.createBinaryVar("", 0.2)
          ilpSolver.addConsBasicLinear(s"", Seq(x, predVar), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"", Seq(x, argumentVar), Seq(1.0, -1.0), None, Some(0.0))
          (pred, arg, x)
        }
      }

      questionParagraphAlignments ++= pVerbQVerbAlignments.toBuffer ++ pVerbQVerbPredicateAlignments.toBuffer
      interParagraphAlignments ++= pPrepVerbAlignments.toBuffer ++ pPrepArgumentAlignments.toBuffer ++ verbSRLEdges.toBuffer

      // constraint: the predicate can be active only if at least one of the its connected arguments are active
      pVerbPredicates.foreach { predicate =>
        val predicateVar = activeParagraphVerbSRLConstituents(predicate)
        val arguments = pVerbPredicateToArgumentMap(predicate)
        val argumentsVars = arguments.map(activeParagraphVerbSRLConstituents)
        val weights = arguments.map(_ => 1.0)
        ilpSolver.addConsBasicLinear(s"", argumentsVars :+ predicateVar, weights :+ -1.0, Some(0.0), None)
      }

      // PA alignments: alignment between verb-srl argument in paragraph and answer option
      val ansPVerbAlignments = for {
        verbC <- pVerbArguments
        (ansIdx, ansVar) <- activeAnswerOptions
        score = alignmentFunction.scoreCellCell(verbC.getSurfaceForm, q.answers(ansIdx).answerText)
        if score >= 0.70 // TODO: tune this
      } yield {
        val x = ilpSolver.createBinaryVar("", score)

        // constraint: if pairwise variable is active, the two end should be active too.
        val srlVerbVar = activeParagraphVerbSRLConstituents(verbC)
        ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, srlVerbVar), Seq(1.0, -1.0), None, Some(0.0))
        ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, ansVar), Seq(1.0, -1.0), None, Some(0.0))

        // constraint: if the argument is active, the predicate in the same frame should be active too.
        //        val predicate = verbC.getIncomingRelations.get(0).getSource
        //        val predicateVar = activeParagraphVerbSRLConstituents(predicate)
        //        ilpSolver.addConsBasicLinear(s"", Seq(srlVerbVar, predicateVar), Seq(1.0, -1.0), None, Some(0.0))

        // constraint: if the argument is active, one other argument in the same frame should be active too.
        //        val constituents = pVerbPredicateToArgumentMap(predicate).filter(_ != verbC)
        //        val constituentVars = constituents.map(activeParagraphVerbSRLConstituents)
        //        val weights = constituentVars.map(_ => -1.0)
        //        ilpSolver.addConsBasicLinear(s"", srlVerbVar +: constituentVars, 1.0 +: weights, None, Some(0.0))
        (verbC, ansIdx, 0, x)
      }

      // constraint: not more than one argument of a frame, should be aligned to answer option:
      ansPVerbAlignments.groupBy(_._1.getIncomingRelations.get(0).getSource).foreach { case (_, list) =>
        val variables = list.map(_._4)
        val weights = variables.map(_ => 1.0)
        ilpSolver.addConsBasicLinear(s"", variables, weights, None, Some(1.0))
      }

      paragraphAnswerAlignments ++= ansPVerbAlignments.toBuffer

      // constraint: all verb-srl arguments in the paragraphs should have at least two connections.
      def getEdgesConnectedToVerbSRLArguments(c: Constituent): Seq[V] = {
        ansPVerbAlignments.filter(_._1 == c).map(_._4) ++ pVerbQVerbAlignments.filter(_._2 == c).map(_._3) ++ pPrepVerbAlignments.filter(_._1 == c).map(_._3)
      }

      pVerbArguments.foreach { arg =>
        val pVerbArgVar = activeParagraphVerbSRLConstituents(arg)
        val vars = getEdgesConnectedToVerbSRLArguments(arg)
        if (vars.size >= 2) {
          val weights = vars.map(_ => -1.0)
          ilpSolver.addConsBasicLinear(s"", pVerbArgVar +: vars, 2.0 +: weights, None, Some(0.0))
        }
        else {
          // if nothing is connected to it, then it should be inactive
          ilpSolver.addConsBasicLinear(s"", Seq(pVerbArgVar), Seq(1.0), None, Some(0.0))
        }
      }

      // constraint: no dangling arguments: i.e. any verb-srl argument in the paragraph, should be connected to at least
      // one other thing, in addition to its predicate
      def getConstituentsConectedToParagraphSRLArg(c: Constituent): Seq[V] = {
        pPrepVerbAlignments.filter(_._1 == c).map {
          _._3
        } ++ ansPVerbAlignments.filter(_._1 == c).map {
          _._4
        }
      }

      pVerbArguments.zipWithIndex.foreach { case (arg, idx) =>
        val argVar = activeParagraphVerbSRLConstituents(arg)
        val connected = getConstituentsConectedToParagraphSRLArg(arg)
        if (connected.nonEmpty) {
          val weights = connected.map(_ => -1.0)
          ilpSolver.addConsBasicLinear(s"", argVar +: connected, 1.0 +: weights, None, Some(0.0))
        }
        else {
          // if nothing is connected to it, then it should be inactive
          ilpSolver.addConsBasicLinear(s"", Seq(argVar), Seq(1.0), None, Some(0.0))
        }
      }

      // constraint: verb-srl predicates, if active, have to be connected to at least two things
      def getEdgesConnectedToVerbSRLPredicates(c: Constituent): Seq[V] = {
        verbSRLEdges.filter(_._1 == c).map(_._3) ++ pVerbQVerbPredicateAlignments.filter(_._2 == c).map(_._3)
      }

      pVerbPredicates.zipWithIndex.foreach { case (arg, idx) =>
        val argVar = activeParagraphVerbSRLConstituents(arg)
        val connected = getEdgesConnectedToVerbSRLPredicates(arg)
        if (connected.size >= 2) {
          val weights = connected.map(_ => -1.0)
          ilpSolver.addConsBasicLinear(s"", argVar +: connected, 2.0 +: weights, None, Some(0.0))
        }
        else {
          // if nothing is connected to it, then it should be inactive
          ilpSolver.addConsBasicLinear(s"", Seq(argVar), Seq(1.0), None, Some(0.0))
        }
      }

      /*
            def getConnectedEdgesToParagraphVerbPredicate(pred: Constituent): Seq[V] = {
              val pVerbQVerbAlignments1 = pVerbQVerbAlignments.filter(_._2 == pred)
              //val verbSRLEdges1 = verbSRLEdges.filter(_._1 == pred)
              /*verbSRLEdges1.map(_._3) ++*/ pVerbQVerbAlignments1.map(_._3)
            }
      */

      /*      pVerbPredicates.foreach{ p =>
              // constraint:
              // for each predicate in the paragraph, it should not be active, unless it has at least one verb connected to it from question
              // 2 * pVar  <= \sum edges
              val edges = getConnectedEdgesToParagraphVerbPredicate(p)
              val pVar = activeParagraphVerbSRLConstituents(p)
              val weights = edges.map(_ => -1.0)
              ilpSolver.addConsBasicLinear(s"", pVar +: edges, 1.0 +: weights, None, Some(0.0))
            }*/

    }

    if (reasoningTypes.contains(VerbSRLandCoref)) {
      val qVerbViewOpt = if (qTA.hasView(srlViewName)) Some(qTA.getView(srlViewName)) else None
      val qVerbConstituents = if (qVerbViewOpt.isDefined) qVerbViewOpt.get.getConstituents.asScala else Seq.empty
      val pVerbConstituents = if (pTA.hasView(srlViewName)) pTA.getView(srlViewName).getConstituents.asScala else Seq.empty
      val qVerbPredicates = qVerbConstituents.filter(_.getLabel == "Predicate")
      val pVerbPredicates = pVerbConstituents.filter(_.getLabel == "Predicate")
      val qVerbArguments = qVerbConstituents diff qVerbPredicates
      val pVerbArguments = pVerbConstituents diff pVerbPredicates
      val pCorefConstituents = if (pTA.hasView(TextILPSolver.stanfordCorefViewName)) pTA.getView(TextILPSolver.stanfordCorefViewName).getConstituents.asScala else Seq.empty
      val pCorefConstituentGroups = pCorefConstituents.groupBy(_.getLabel)

      val pVerbPredicateToArgumentMap = pVerbConstituents.map(c => c -> c.getOutgoingRelations.asScala.map(_.getTarget)).toMap
      val qVerbPredicateToArgumentMap = qVerbConstituents.map(c => c -> c.getOutgoingRelations.asScala.map(_.getTarget)).toMap

      // active paragraph comma-srl constituents
      val activeParagraphCorefConstituents = pCorefConstituents.zipWithIndex.map { case (c, idx) =>
        // TODO: tune this weight
        c -> ilpSolver.createBinaryVar(s"activeParagraphCorefCons=$idx", 0.001)
      }.toMap

      // active question verb-srl constituents
      val activeQuestionVerbSRLConstituents = qVerbConstituents.zipWithIndex.map { case (c, idx) =>
        // TODO: tune this weight
        c -> ilpSolver.createBinaryVar(s"activeQuestionVerbCons=$idx", 0.001)
      }.toMap

      // active paragraph verb-srl constituents
      val activeParagraphVerbSRLConstituents = pVerbConstituents.zipWithIndex.map { case (c, idx) =>
        // TODO: tune this weight
        c -> ilpSolver.createBinaryVar(s"activeParagraphVerbCons=$idx", 0.001)
      }.toMap

      val activeCorefChains = pCorefConstituentGroups.keySet.map { key =>
        val x = ilpSolver.createBinaryVar(s"", -0.0001)
        // constraint: if a coref cons is active, then the chain must be active
        val consInChain = pCorefConstituentGroups(key)
        val consVars = consInChain.map(activeParagraphCorefConstituents)
        consVars.foreach { cVar =>
          ilpSolver.addConsBasicLinear(s"", Seq(cVar, x), Seq(1.0, -1.0), None, Some(0.0))
        }
        // constraint: if the coref chain is active, at least one of the constituents that belong to that chain should be active
        ilpSolver.addConsBasicLinear(s"", x +: consVars, 1.0 +: consVars.map(_ => -1.0), None, Some(0.0))
        key -> x
      }.toMap

      /*
      // find index of "if" and then find the first NP after that.
      val selectedCons = if(qVerbViewOpt.isDefined && qTA.text.contains(" if ")) {
        val patternOpt = qTA.getView(ViewNames.SHALLOW_PARSE).getConstituents.asScala.sliding(2).find{ list =>
          val c1 = list(0)
          val c2 = list(1)
          c1.getSurfaceForm == "if" && (c2.getLabel == "NP" || c2.getLabel == "VP")
        }
        if(patternOpt.isDefined) {
          val c1 = patternOpt.get(1)
          val c0 = patternOpt.get(0)
          qVerbViewOpt.get.
            getConstituentsOverlappingCharSpan(c1.getStartCharOffset, c1.getEndCharOffset).asScala.
            filter{ _.getStartSpan >= c0.getEndSpan }
        }
        else {
          Seq.empty
        }
      }
      else {
        Seq.empty
      }

      println("Selected: " + selectedCons)

      // if at least one cons is selected, at least k of these should
      if(selectedCons.nonEmpty) {
        val selectedVars = selectedCons.map(activeQuestionVerbSRLConstituents)
        ilpSolver.addConsAtLeastOne(s"", selectedVars) //TODO: tune this
      }
      */

      // constraint: have at most 1 coref chain
      ilpSolver.addConsAtMostK(s"", activeCorefChains.values.toSeq, 2) //TODO: tune this

      // constrain: have at least one coref constituents
      ilpSolver.addConsAtLeastOne(s"", activeParagraphCorefConstituents.values.toSeq) //TODO: tune this

      // constraint: have some constituents used from the question
      ilpSolver.addConsAtLeastK(s"", activeQuestionVerbSRLConstituents.values.toSeq, 2.0)

      // constraint: have at most k active srl-verb predicates in the paragraph
      val predicateVariables = pVerbPredicates.map(activeParagraphVerbSRLConstituents)
      ilpSolver.addConsAtMostK(s"", predicateVariables, 1.0) // TODO: tune this number

      // PP alignments: alignment between paragraph coref cons and paragraph verb-srl argument
      val pCorefConsPVerbAlignments = for {
        verbC <- pVerbArguments
        corefC <- pCorefConstituents
        score = TextILPSolver.sahandClient.getScore(verbC.getSurfaceForm, corefC.getSurfaceForm, SimilarityNames.phrasesim)
        if score >= 0.75 // TODO: tune this
      }
        yield {
          val x = ilpSolver.createBinaryVar("", score)
          // constraint: if pairwise variable is active, the two end should be active too.
          val srlVerbVar = activeParagraphVerbSRLConstituents(verbC)
          val corefVar = activeParagraphCorefConstituents(corefC)
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, srlVerbVar), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, corefVar), Seq(1.0, -1.0), None, Some(0.0))
          (verbC, corefC, x)
        }

      // PP: coref alignments
      // there is alignment between coref constituents that are in the same cluster
      val pCorefArgumentAlignments = for {
        chainIdx <- pCorefConstituentGroups.keySet
        consInChain = pCorefConstituentGroups(chainIdx)
        subsetsOfSize2 = consInChain.combinations(2)
        pair <- subsetsOfSize2
        corefCons1 = pair(0)
        corefCons2 = pair(1)
      }
        yield {
          val x = ilpSolver.createBinaryVar("", 0.02)
          // TODO: tune this
          // constraint: if pairwise variable is active, the two end should be active too.
          val arg1Var = activeParagraphCorefConstituents(corefCons1)
          val arg2Var = activeParagraphCorefConstituents(corefCons2)
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, arg1Var), Seq(1.0, -1.0), Some(0.0), Some(0.0))
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, arg2Var), Seq(1.0, -1.0), Some(0.0), Some(0.0))
          (corefCons1, corefCons2, x)
        }

      // if the coref constituent is active, at least one of the edges connected to it should be active
      def getEdgesConnectedToCons(c: Constituent): Set[V] = {
        pCorefArgumentAlignments.filter(_._1 == c).map(_._3) ++
          pCorefArgumentAlignments.filter(_._2 == c).map(_._3) ++
          pCorefConsPVerbAlignments.filter(_._2 == c).map(_._3)
      }

      activeParagraphCorefConstituents.foreach { case (c, x) =>
        val edgeVars = getEdgesConnectedToCons(c).toSeq
        ilpSolver.addConsBasicLinear("", x +: edgeVars, 1.0 +: edgeVars.map(_ => -1.0), None, Some(0.0))
      }

      // QP alignments: alignment between question verb-srl argument paragraph coref constituents
      val pCorefConsQVerbAlignments = {
        for {
          verbC <- qVerbArguments
          corefC <- pCorefConstituents
          score = alignmentFunction.scoreCellQCons(verbC.getSurfaceForm, corefC.getSurfaceForm)
          if score >= 0.60 //TODO: tune this
        } yield {
          val x = ilpSolver.createBinaryVar("", score)
          // constraint: if pairwise variable is active, the two end should be active too.
          val srlVerbVar = activeQuestionVerbSRLConstituents(verbC)
          val corefVar = activeParagraphCorefConstituents(corefC)
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint1", Seq(x, srlVerbVar), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"activeFrameConstraint2", Seq(x, corefVar), Seq(1.0, -1.0), None, Some(0.0))
          (verbC, corefC, x)
        }
      }

      // QP alignments: predicate-predicate
      val pVerbQVerbAlignments = for {
        qVerbC <- qVerbPredicates
        pVerbC <- pVerbPredicates
        score = TextILPSolver.sahandClient.getScore(pVerbC.getSurfaceForm,
          qVerbC.getSurfaceForm, SimilarityNames.phrasesim)
        //alignmentFunction.scoreCellCell(pVerbC.getSurfaceForm, qVerbC.getSurfaceForm)
        if score >= 0.4 // TODO: tune this
      } yield {
        val x = ilpSolver.createBinaryVar("", score)
        // constraint: if pairwise variable is active, the two end should be active too.
        val qVerbVar = activeQuestionVerbSRLConstituents(qVerbC)
        val pVerbVar = activeParagraphVerbSRLConstituents(pVerbC)
        ilpSolver.addConsBasicLinear(s"", Seq(x, qVerbVar), Seq(1.0, -1.0), None, Some(0.0))
        ilpSolver.addConsBasicLinear(s"", Seq(x, pVerbVar), Seq(1.0, -1.0), None, Some(0.0))
        (qVerbC, pVerbC, x)
      }

      // constraint: question cons can be active, only if any least one of the edges connected to it is active
      def getEdgesConnectedToQuestionCons(c: Constituent): Seq[V] = {
        pVerbQVerbAlignments.filter(_._1 == c).map(_._3) ++ pCorefConsQVerbAlignments.filter(_._1 == c).map(_._3)
      }

      activeQuestionVerbSRLConstituents.foreach { case (c, x) =>
        val edges = getEdgesConnectedToQuestionCons(c)
        val weights = edges.map(_ => -1.0)
        ilpSolver.addConsBasicLinear(s"", x +: edges, 1.0 +: weights, None, Some(0.0))
      }

      // constraint: predicate should be inactive, unless it has at least two incoming outgoing relations
      // pred * 2 <= \sum arguments
      val verbSRLEdges = pVerbPredicates.flatMap { pred =>
        val arguments = pVerbPredicateToArgumentMap(pred)
        val predVar = activeParagraphVerbSRLConstituents(pred)
        val argumentVars = arguments.map(activeParagraphVerbSRLConstituents)
        val weights = argumentVars.map(_ => -1.0)
        //ilpSolver.addConsBasicLinear(s"", predVar +: argumentVars, 2.0 +: weights, None, Some(0.0))

        // if predicate is inactive, nothing can be active
        argumentVars.foreach { v =>
          ilpSolver.addConsBasicLinear(s"", Seq(v, predVar), Seq(1.0, -1.0), None, Some(0.0))
        }
        //ilpSolver.addConsBasicLinear(s"", predVar +: argumentVars, 10.0 +: weights, Some(0.0), None)

        // get variables for SRL edges
        arguments.map { arg =>
          val argumentVar = activeParagraphVerbSRLConstituents(arg)
          // if both edges are active, visualize an edge
          val x = ilpSolver.createBinaryVar("", 0.01)
          ilpSolver.addConsBasicLinear(s"", Seq(x, predVar), Seq(1.0, -1.0), None, Some(0.0))
          ilpSolver.addConsBasicLinear(s"", Seq(x, argumentVar), Seq(1.0, -1.0), None, Some(0.0))
          (pred, arg, x)
        }
      }

      questionParagraphAlignments ++= pCorefConsQVerbAlignments.toBuffer ++ pVerbQVerbAlignments.toBuffer
      interParagraphAlignments ++= pCorefConsPVerbAlignments.toBuffer ++ pCorefArgumentAlignments.toBuffer ++ verbSRLEdges.toBuffer

      // constraint: the predicate can be active only if at least one of the its connected arguments are active
      pVerbPredicates.foreach { predicate =>
        val predicateVar = activeParagraphVerbSRLConstituents(predicate)
        val arguments = pVerbPredicateToArgumentMap(predicate)
        val argumentsVars = arguments.map(activeParagraphVerbSRLConstituents)
        val weights = arguments.map(_ => 1.0)
        ilpSolver.addConsBasicLinear(s"", argumentsVars :+ predicateVar, weights :+ -1.0, Some(0.0), None)
      }

      // PA alignments: alignment between verb-srl argument in paragraph and answer option
      val ansPVerbAlignments = for {
        verbC <- pVerbArguments
        (ansIdx, ansVar) <- activeAnswerOptions
        score = alignmentFunction.scoreCellCell(verbC.getSurfaceForm, q.answers(ansIdx).answerText)
        if score >= 0.65 // TODO: tune this
      } yield {
        val x = ilpSolver.createBinaryVar("", score)

        // constraint: if pairwise variable is active, the two end should be active too.
        val srlVerbVar = activeParagraphVerbSRLConstituents(verbC)
        ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, srlVerbVar), Seq(1.0, -1.0), None, Some(0.0))
        ilpSolver.addConsBasicLinear(s"activeFrameConstraint", Seq(x, ansVar), Seq(1.0, -1.0), None, Some(0.0))

        // constraint: if the argument is active, the predicate in the same frame should be active too.
        //        val predicate = verbC.getIncomingRelations.get(0).getSource
        //        val predicateVar = activeParagraphVerbSRLConstituents(predicate)
        //        ilpSolver.addConsBasicLinear(s"", Seq(srlVerbVar, predicateVar), Seq(1.0, -1.0), None, Some(0.0))

        // constraint: if the argument is active, one other argument in the same frame should be active too.
        //        val constituents = pVerbPredicateToArgumentMap(predicate).filter(_ != verbC)
        //        val constituentVars = constituents.map(activeParagraphVerbSRLConstituents)
        //        val weights = constituentVars.map(_ => -1.0)
        //        ilpSolver.addConsBasicLinear(s"", srlVerbVar +: constituentVars, 1.0 +: weights, None, Some(0.0))
        (verbC, ansIdx, 0, x)
      }

      // constraint: not more than one argument of a frame, should be aligned to answer option:
      ansPVerbAlignments.groupBy(_._1.getIncomingRelations.get(0).getSource).foreach { case (_, list) =>
        val variables = list.map(_._4)
        val weights = variables.map(_ => 1.0)
        ilpSolver.addConsBasicLinear(s"", variables, weights, None, Some(1.0))
      }

      paragraphAnswerAlignments ++= ansPVerbAlignments.toBuffer

      def getConstituentsConectedToParagraphSRLArg(c: Constituent): Seq[V] = {
        pCorefConsPVerbAlignments.filter(_._1 == c).map {
          _._3
        } ++ ansPVerbAlignments.filter(_._1 == c).map {
          _._4
        }
      }

      // constraint: no dangling arguments: i.e. any verb-srl argument in the paragram, should be connected to at least
      // one other thing, in addition to its predicate
      pVerbArguments.zipWithIndex.foreach { case (arg, idx) =>
        val argVar = activeParagraphVerbSRLConstituents(arg)
        val connected = getConstituentsConectedToParagraphSRLArg(arg)
        if (connected.nonEmpty) {
          val weights = connected.map(_ => -1.0)
          ilpSolver.addConsBasicLinear(s"", argVar +: connected, 1.0 +: weights, None, Some(0.0))
        }
        else {
          // if nothing is connected to it, then it should be inactive
          ilpSolver.addConsBasicLinear(s"", Seq(argVar), Seq(1.0), None, Some(0.0))
        }
      }

      def getConnectedEdgesToParagraphVerbPredicate(pred: Constituent): Seq[V] = {
        val pVerbQVerbAlignments1 = pVerbQVerbAlignments.filter(_._2 == pred)
        //val verbSRLEdges1 = verbSRLEdges.filter(_._1 == pred)
        //println("pVErbAlignments1: " + pVerbQVerbAlignments1)
        //println("verbSRLEdges1: " + verbSRLEdges1)
        /*verbSRLEdges1.map(_._3) ++*/ pVerbQVerbAlignments1.map(_._3)
      }

      pVerbPredicates.foreach { p =>
        // constraint:
        // for each predicate in the paragraph, it should not be active, unless it has at least one verb connected to it from question
        // 2 * pVar  <= \sum edges
        val edges = getConnectedEdgesToParagraphVerbPredicate(p)
        val pVar = activeParagraphVerbSRLConstituents(p)
        val weights = edges.map(_ => -1.0)
        ilpSolver.addConsBasicLinear(s"", pVar +: edges, 1.0 +: weights, None, Some(0.0))
      }

      // no dangling coref cons
      def getConnectedEdgesToPCorefCons(corefC: Constituent): Seq[V] = {
        pCorefConsPVerbAlignments.filter(_._2 == corefC).map(_._3)
      }

      pCorefConstituents.foreach { c =>
        val connected = getConnectedEdgesToPCorefCons(c)
        val weights = connected.map(_ => -1.0)
        // it is active, only if something connected to it is also active
        val cVar = activeParagraphCorefConstituents(c)
        ilpSolver.addConsBasicLinear(s"", cVar +: connected, 1.0 +: weights, None, Some(0.0))
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

  def solveAllAnswerOptions(question: String, options: Seq[String], snippet: String, reasoningType: ReasoningType, srlVu: String): Seq[(Seq[Int], EntityRelationResult)] = {
    val annotationStart = System.currentTimeMillis()
    val (q: Question, p: Paragraph) = preprocessQuestionData(question, options, snippet)
    val annotationEnd = System.currentTimeMillis()

    reasoningType match {
      case SRLV1Rule => Seq(SRLSolverV1(q, p, srlVu))
      case SRLV2Rule => Seq(SRLSolverV2(q, p, srlVu))
      case SRLV3Rule => Seq(SRLSolverV3(q, p, aligner, srlVu))
      case WhatDoesItDoRule => Seq(WhatDoesItDoSolver(q, p))
      case CauseRule => Seq(CauseResultRules(q, p))
      case x =>
        val ilpSolver = new ScipSolver("textILP", ScipParams.Default)

        val modelCreationStart = System.currentTimeMillis()

        val (questionParagraphAlignments, paragraphAnswerAlignments,
        interParagraphAlignments, activeAnswerOptions,
        isTrueFalseQuestion, aTokens) = createILPModel(q, p, ilpSolver, aligner, Set(reasoningType), useSummary = true, srlVu)
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

  def getSolverPredictionsAsWekaInstance(question: String, options: Seq[String], snippet: String): Instances = {
    val fvs = extractFeatureVectorForQuestion(question, options, snippet)
    // convert out feature vector to Weka feature vector
    val dataUnlabeled = new Instances(TextILPSolver.headerEmptyInstances)
    val confidenceValuesPerAnsOption = fvs.foreach { featureVector =>
      val ins = new SparseInstance(featureVector.length)
      featureVector.zipWithIndex.foreach { case (value, i) => ins.setValue(i, value) }
      dataUnlabeled.add(ins)
    }
    dataUnlabeled
  }

  def predictMaxScoreWekaClassifier(question: String, options: Seq[String], snippet: String): Int = {
    val (confidenceValuesPerAnsOption, _) = predictAllCandidatesWithWekaClassifier(question, options, snippet)
    confidenceValuesPerAnsOption.zipWithIndex.maxBy(_._1)._2
  }

  def predictAllCandidatesWithWekaClassifier(question: String, options: Seq[String], snippet: String): (Seq[Double], Instances) = {
    // note there is inefficienss here. The instance gets pre-processed for each reasoning combination inside "distributionForInstance"
    val fvs = extractFeatureVectorForQuestion(question, options, snippet)
    val dataUnlabeled = new Instances(TextILPSolver.headerEmptyInstances)
    dataUnlabeled.setClassIndex(dataUnlabeled.numAttributes() - 1)
    val confidenceValuesPerAnsOption = fvs.zipWithIndex.map { case (featureVector, i) =>
      val ins = new DenseInstance(featureVector.length)
      ins.replaceMissingValues(featureVector.toArray)
      dataUnlabeled.add(ins)
      TextILPSolver.wekaClassifier.distributionForInstance(dataUnlabeled.instance(i))(0)
    }
    (confidenceValuesPerAnsOption, dataUnlabeled)
  }

  def extractFeatureVectorForQuestionWithCorrectLabel(question: String, options: Seq[String], snippet: String, correctIdx: Int): Seq[Seq[Double]] = {
    val vectors = extractFeatureVectorForQuestion(question, options, snippet)
    vectors.zipWithIndex.map { case (featureVector, idx) =>
      featureVector :+ (if (idx == correctIdx) 1.0 else 0.0)
    }
  }

  private def extractFeatureVectorForQuestion(question: String, options: Seq[String], snippet: String) = {
    val types = Constants.textILPModel match {
      case TextILPModel.EnsembleFull | TextILPModel.EnsembleMinimal =>
        Seq(SimpleMatching, WhatDoesItDoRule, CauseRule, SRLV1Rule, VerbSRLandPrepSRL, SRLV1ILP, VerbSRLandCoref, SRLV2Rule, SRLV3Rule, VerbSRLandCommaSRL)
      case TextILPModel.EnsembleNoCoref =>
        Seq(SimpleMatching, WhatDoesItDoRule, CauseRule, SRLV1Rule, VerbSRLandPrepSRL, SRLV1ILP, SRLV2Rule, SRLV3Rule, VerbSRLandCommaSRL)
      case TextILPModel.EnsembleNoPrepSRL =>
        Seq(SimpleMatching, WhatDoesItDoRule, CauseRule, SRLV1Rule, SRLV1ILP, VerbSRLandCoref, SRLV2Rule, SRLV3Rule, VerbSRLandCommaSRL)
      case TextILPModel.EnsembleNoCommaSRL =>
        Seq(SimpleMatching, WhatDoesItDoRule, CauseRule, SRLV1Rule, VerbSRLandPrepSRL, SRLV1ILP, VerbSRLandCoref, SRLV2Rule, SRLV3Rule)
      case TextILPModel.EnsembleNoSimpleMatching =>
        Seq(WhatDoesItDoRule, CauseRule, SRLV1Rule, VerbSRLandPrepSRL, SRLV1ILP, VerbSRLandCoref, SRLV2Rule, SRLV3Rule, VerbSRLandCommaSRL)
      case TextILPModel.EnsembleNoVerbSRL =>
        Seq(SimpleMatching, WhatDoesItDoRule, CauseRule, VerbSRLandPrepSRL, VerbSRLandCoref, VerbSRLandCommaSRL)
      case default =>
        throw new Exception(s"Feature extraction is not defined for your selection ${Constants.textILPModel}")
    }
    val srlViewsAll = if (Constants.textILPModel == TextILPModel.EnsembleMinimal) {
      Seq(ViewNames.SRL_VERB, TextILPSolver.pathLSTMViewName)
    }
    else {
      Seq(ViewNames.SRL_VERB, TextILPSolver.curatorSRLViewName, TextILPSolver.pathLSTMViewName)
    }

    val results = types.flatMap { t =>
      val srlViewTypes = if (t == CauseRule || t == WhatDoesItDoRule || t == SimpleMatching) Seq(TextILPSolver.pathLSTMViewName) else srlViewsAll
      srlViewTypes.flatMap { srlVu =>
        val results = solveAllAnswerOptions(question, options, snippet, t, srlVu)
        results.flatMap { case (a, b) =>
          a.map { c => (t, srlVu, c) -> b.statistics }
        }
      }
    }.toMap

    options.indices.map { idx =>
      types.flatMap { t =>
        val srlViewTypes = if (t == CauseRule || t == WhatDoesItDoRule || t == SimpleMatching) Seq(TextILPSolver.pathLSTMViewName) else srlViewsAll
        srlViewTypes.flatMap { srlVu =>
          if (results.contains((t, srlVu, idx))) {
            val result = results((t, srlVu, idx))
            val selected = if (result.selected) 1.0 else 0.0
            val binVarNum = result.numberOfBinaryVars
            val constNum = result.numberOfConstraints
            val objValue = result.objectiveValue
            Seq(selected, objValue, binVarNum, constNum, if (binVarNum == 0.0) 0.0 else objValue / binVarNum, if (constNum == 0.0) 0.0 else objValue / constNum)
          }
          else {
            Seq(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
          }
        }
      }
    }
  }
}
