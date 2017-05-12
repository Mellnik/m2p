#ifndef __DATAPOINT_H
#define __DATAPOINT_H

class DataPoint
{
public:
	DataPoint();
	~DataPoint();
	
	int m_Date;
	double m_High;
	double m_Low;
	double m_Open;
	double m_Close;
	double m_Volume;
	double m_CoinVolume;
	double m_WeightedAverage;

private:
};

#endif /* __DATAPOINT_H */