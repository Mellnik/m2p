#ifndef __NEURALNETWORK_H
#define __NEURALNETWORK_H

#include "DataPeriod.h"

class NeuralNetwork
{
public:
	NeuralNetwork();
	~NeuralNetwork();

	void SetNumberOfPeriods(size_t Size);

	bool LoadMarketTradingFromFile(string MarketTradingFile, int DataPointTime, int DataPointsPerPeriod);

	void Debug();

private:
	vector<DataPeriod *> m_Periods;
};

#endif /* __NEURALNETWORK_H */