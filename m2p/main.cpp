#include "main.h"
#include "NeuralNetwork.h"

/*
PS D:\Local> .\wget.exe "https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=1494525600&end=1494554400&period=300" -O BTC_ETH_8hrs.json // 96 entries
PS D:\Local> .\wget.exe "https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=1493316209&end=9999999999&period=300" -O BTC_ETH_2weeks.json
PS D:\Local> .\wget.exe "https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=1462989809&end=9999999999&period=300" -O BTC_ETH_1year.json
*/

int main(int argc, char *argv[])
{
	cout << PROGRAMSTART << endl;

	NeuralNetwork *Instance = new NeuralNetwork();
	
	// 300 seconds = 1 candle capture (DataPoint)
	// 72 candle captures (DataPoints) per DataPeriod
	// 2 weeks = 4032 DataPoints / 72 = 56 DataPeriods
	if (!Instance->LoadMarketTradingFromFile("C:\\Users\\wendelstein\\Documents\\Visual Studio 2015\\Projects\\m2p\\BTC_ETH_2weeks.json", 300, 72))
	{
		return 1;
	}
	
	Instance->Debug(1);

	delete Instance;

	/*ifstream i("D:\\Local\\BTC_ETH_2weeks.json");
	json j;
	i >> j;

	cout << "Read " << j.size() << " trading data points." << endl;

	for (json::iterator it = j.begin(); it != j.end(); ++it)
	{
		cout << it.value().at("close") << endl;
		break;
	}

	ofstream o("D:\\Local\\pretty.json");
	o << std::setw(4) << j << endl;
	cout << "Written." << endl;
	o.close();*/

	cin >> argc;
	return 0;
}