package SVM

import (
	"fmt"
	"math"
	"math/rand"
)

// LinearSVM представляет линейный классификатор опорных векторов
type LinearSVM struct {
	Weights         []float64
	Bias            float64
	Classes         []string
	ClassMap        map[string]int
	FeatureDim      int
	SVMLearningRate float64
	SVMLambda       float64
	SVMTolerance    float64
	SVMEpochs       int
}

// NewLinearSVM создает новый SVM классификатор
func NewLinearSVM(classes []string, featureDim int, SVMLearningRate, SVMLambda, SVMTolerance float64, SVMEpochs int) *LinearSVM {
	classMap := make(map[string]int)
	for i, class := range classes {
		classMap[class] = i
	}

	return &LinearSVM{
		Weights:         make([]float64, featureDim*len(classes)),
		Bias:            0,
		Classes:         classes,
		ClassMap:        classMap,
		FeatureDim:      featureDim,
		SVMEpochs:       SVMEpochs,
		SVMLearningRate: SVMLearningRate,
		SVMLambda:       SVMLambda,
		SVMTolerance:    SVMTolerance,
	}
}

// Train обучает SVM классификатор
func (svm *LinearSVM) Train(X [][]float64, y []string) {
	nClasses := len(svm.Classes)
	nFeatures := svm.FeatureDim

	// Инициализация весов
	for i := range svm.Weights {
		svm.Weights[i] = 0.01 * (2*rand.Float64() - 1)
	}

	// Преобразуем метки в числовые
	yNumeric := make([]int, len(y))
	for i, label := range y {
		yNumeric[i] = svm.ClassMap[label]
	}

	// Обучение с использованием стохастического градиентного спуска
	prevLoss := math.Inf(1)
	for epoch := 0; epoch < svm.SVMEpochs; epoch++ {
		loss := 0.0

		// Случайное перемешивание данных
		indices := make([]int, len(X))
		for i := range indices {
			indices[i] = i
		}

		// Стохастический градиентный спуск
		for _, idx := range indices {
			x := X[idx]
			trueClass := yNumeric[idx]

			// Вычисляем предсказания для всех классов
			scores := make([]float64, nClasses)
			for c := 0; c < nClasses; c++ {
				score := svm.Bias
				for f := 0; f < nFeatures; f++ {
					score += svm.Weights[c*nFeatures+f] * x[f]
				}
				scores[c] = score
			}

			// Находим класс с максимальным счетом
			maxScore := scores[trueClass]
			maxScoreIdx := trueClass
			for c := 0; c < nClasses; c++ {
				if c != trueClass && scores[c] > maxScore {
					maxScore = scores[c]
					maxScoreIdx = c
				}
			}

			// Вычисляем функцию потерь (hinge loss)
			margin := maxScore - scores[trueClass] + 1.0
			if margin > 0 {
				loss += margin

				// Обновляем веса
				for f := 0; f < nFeatures; f++ {
					// Градиент для истинного класса
					svm.Weights[trueClass*nFeatures+f] += svm.SVMLearningRate * x[f]
					// Градиент для неправильного класса
					svm.Weights[maxScoreIdx*nFeatures+f] -= svm.SVMLearningRate * x[f]
					// Регуляризация
					svm.Weights[trueClass*nFeatures+f] -= svm.SVMLearningRate * svm.SVMLambda * svm.Weights[trueClass*nFeatures+f]
					svm.Weights[maxScoreIdx*nFeatures+f] -= svm.SVMLearningRate * svm.SVMLambda * svm.Weights[maxScoreIdx*nFeatures+f]
				}
				svm.Bias += svm.SVMLearningRate
				svm.Bias -= svm.SVMLearningRate * svm.SVMLambda * svm.Bias
			} else {
				// Регуляризация даже когда потери нет
				for f := 0; f < nFeatures; f++ {
					for c := 0; c < nClasses; c++ {
						svm.Weights[c*nFeatures+f] -= svm.SVMLearningRate * svm.SVMLambda * svm.Weights[c*nFeatures+f]
					}
				}
				svm.Bias -= svm.SVMLearningRate * svm.SVMLambda * svm.Bias
			}
		}

		loss /= float64(len(X))

		// Проверка сходимости
		if math.Abs(prevLoss-loss) < svm.SVMTolerance {
			fmt.Printf("SVM сошелся на эпохе %d (потеря: %.6f)\n", epoch, loss)
			break
		}
		prevLoss = loss

		if epoch%100 == 0 {
			fmt.Printf("Эпоха %d: потеря = %.6f\n", epoch, loss)
		}
	}
}

// Predict предсказывает класс для одного примера
func (svm *LinearSVM) Predict(x []float64) string {
	nClasses := len(svm.Classes)
	nFeatures := svm.FeatureDim

	maxScore := math.Inf(-1)
	bestClass := ""

	for c := 0; c < nClasses; c++ {
		score := svm.Bias
		for f := 0; f < nFeatures; f++ {
			score += svm.Weights[c*nFeatures+f] * x[f]
		}

		if score > maxScore {
			maxScore = score
			bestClass = svm.Classes[c]
		}
	}

	return bestClass
}

// PredictBatch предсказывает классы для множества примеров
func (svm *LinearSVM) PredictBatch(X [][]float64) []string {
	predictions := make([]string, len(X))
	for i, x := range X {
		predictions[i] = svm.Predict(x)
	}
	return predictions
}
