#ifndef __DATAPERIOD_H
#define __DATAPERIOD_H

#include "DataPoint.h"

class DataPeriod
{
public:
	DataPeriod(size_t Size);
	~DataPeriod();

	void AddDataPoint(DataPoint *point);
	bool IsFull();
	size_t GetSize() const
	{
		return m_Points.size();
	}

	vector<DataPoint *> &GetPoints()
	{
		return m_Points;
	}

private:
	size_t m_Size;

	vector<DataPoint *> m_Points;
};

#endif /* __DATAPERIOD_H */