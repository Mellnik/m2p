#include "main.h"
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
{

}

NeuralNetwork::~NeuralNetwork()
{

}

void NeuralNetwork::SetNumberOfPeriods(size_t Size)
{
	m_Periods.reserve(Size);
}

bool NeuralNetwork::LoadMarketTradingFromFile(string MarketTradingFile, int DataPointTime, int DataPointsPerPeriod)
{
	ifstream TradingData(MarketTradingFile);

	if (!TradingData.is_open())
	{
		Utility::Log("File not existing.");
		return false;
	}

	json j;
	TradingData >> j;
	TradingData.close();

	if (j.size() < 1)
	{
		Utility::Log("File does not contain valid data.");		
		return false;
	}

	if (j.size() % DataPointsPerPeriod != 0)
	{
		Utility::Log("Shit lol");
		return false;
	}

	cout << "Read " << j.size() << " Trading Data Points" << endl;

	size_t NumberOfPeriods = j.size() / DataPointsPerPeriod;
	SetNumberOfPeriods(NumberOfPeriods);
	
	DataPeriod *Period = nullptr;
	for (json::iterator it = j.begin(); it != j.end(); ++it)
	{
		auto Data = it.value();

		if (Period == nullptr)
		{
			Period = new DataPeriod(DataPointsPerPeriod);
		}

		DataPoint *Point = new DataPoint();
		Point->m_Date = Data["date"];
		Point->m_High = Data["high"];
		Point->m_Low = Data["low"];
		Point->m_Open = Data["open"];
		Point->m_Close = Data["close"];
		Point->m_Volume = Data["volume"];
		Point->m_CoinVolume = Data["quoteVolume"];
		Point->m_WeightedAverage = Data["weightedAverage"];

		Period->AddDataPoint(Point);

		if (Period->IsFull())
		{
			m_Periods.push_back(Period);
			Period = nullptr;
		}
	}

	cout << "Set " << NumberOfPeriods << " periods with each " << DataPointsPerPeriod << " data points." << endl;
	return true;
}

void NeuralNetwork::Debug()
{
	int PeriodCount = 0;
	for (auto Period : m_Periods)
	{
		cout << "Period " << ++PeriodCount << " contains " << Period->GetSize() << " Points" << endl;
		int PointCount = 0;
		for (auto Point : Period->GetPoints())
		{
			cout << " Point " << ++PointCount << ":" << endl;
			cout.precision(std::numeric_limits<double>::max_digits10);
			cout << "    Date " << Point->m_Date << ", Open " << std::fixed << Point->m_Open << ", Close " << std::fixed << Point->m_Close << endl;
		}
	}
}