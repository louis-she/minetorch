<template>
  <div class="train">
    <el-container>
      <el-aside width="200px">
        <ul>
          <li>
            <p class="title">Title</p>
            <p class="info">description</p>
            <p class="info">description</p>
          </li>
        </ul>
      </el-aside>
      <el-main>
        <el-tabs v-model="active" @tab-click="handleClick">
          <el-tab-pane label="Log" name="log">
            <log :attach="attach" />
          </el-tab-pane>
          <el-tab-pane label="Graph" name="graph">
            <chart
              :option="chartOption"
            />
          </el-tab-pane>
        </el-tabs>
      </el-main>
    </el-container>
  </div>
</template>

<script>
import log from 'components/log-com'
import chart from 'components/chart'
import Moment from 'moment'

export default {
  components: {
    log,
    chart
  },
  data () {
    return {
      active: 'graph',
      experimentId: '',
      index: 0,
      data1: {
        label: [],
        value: []
      },
      data2: {
        label: [],
        value: []
      },
      // eachart 配置
      chartOption: {
        title: {
          text: '雨量流量关系图',
          subtext: '数据来自西安兰特水电测控技术有限公司',
          x: 'center'
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            animation: false
          }
        },
        legend: {
          data: ['流量', '降雨量'],
          x: 'left'
        },
        toolbox: {
          feature: {
            saveAsImage: {
              title: '保存'
            }
          }
        },
        axisPointer: {
          link: {
            xAxisIndex: 'all'
          }
        },
        dataZoom: [
          {
            show: true,
            realtime: true,
            xAxisIndex: [0, 1]
          },
          {
            type: 'inside',
            realtime: true,
            xAxisIndex: [0, 1]
          }
        ],
        grid: [
          {
            left: 50,
            right: 50,
            height: '35%'
          },
          {
            left: 50,
            right: 50,
            top: '55%',
            height: '35%'
          }
        ],
        xAxis: [
          {
            type: 'category',
            boundaryGap: false,
            axisLine: {onZero: true},
            data: []
          },
          {
            gridIndex: 1,
            type: 'category',
            boundaryGap: false,
            axisLine: {onZero: true},
            data: [],
            position: 'top'
          }
        ],
        yAxis: [
          {
            name: '流量(m^3/s)',
            type: 'value'
          },
          {
            gridIndex: 1,
            name: '降雨量(mm)',
            type: 'value',
            inverse: true
          }
        ],
        series: [
          {
            name: '流量',
            type: 'line',
            symbolSize: 8,
            hoverAnimation: false,
            data: []
          },
          {
            name: '降雨量',
            type: 'line',
            xAxisIndex: 1,
            yAxisIndex: 1,
            symbolSize: 8,
            hoverAnimation: false,
            data: []
          }
        ]
      }
    }
  },
  computed: {
    attach () {
      return `http://${process.env.SERVER_ADDR}:${process.env.WEB_SOCKET_PORT}/common?experiment_id=${this.experimentId}`
    }
  },
  mounted () {
    setInterval(() => {
      this.randomData1()
      this.randomData2()
      this.chartOption = {
        xAxis: [
          {
            data: this.data1.label
          },
          {
            data: this.data2.label
          }
        ],
        series: [
          {
            data: this.data1.value
          },
          {
            data: this.data2.value
          }
        ]
      }
    }, 1000)

    this.experimentId = this.$route.params.experimentId
  },
  methods: {
    randomData1 () {
      const value = Math.random() * 1000 + Math.random() * 21 - 10

      this.data1.label.push(Moment().format('HH:mm:ss'))
      this.data1.value.push(Math.round(value))
    },
    randomData2 () {
      let now = +new Date(1997, 9, 3)
      let oneDay = 24 * 3600 * 1000
      let value = Math.random() * 1000

      now = new Date(+now + oneDay)
      value = value + Math.random() * 21 - 10
      this.data2.label.push(this.index++)
      this.data2.value.push(Math.round(value))
    }
  }
}
</script>

<style lang="scss">
.train {
  background-color: #fff;
  border: 1px solid #ebebeb;
  border-radius: 3px;
  .el-aside {
    border-right: 1px solid #ebebeb;
    ul {
      li {
        padding: 10px 20px;
        border-bottom: .46rem solid #f1f1f1;
        background: linear-gradient(to right,#39aa56 0,#39aa56 8px,#fff 8px,#fff 100%) no-repeat;
        .title {
          margin-bottom: 20px;
          font-size: 16px;
          color: #333;
        }
        .info {
          font-size: 14px;
          color: #666;
        }
      }
    }
  }
  .el-main {
    padding: 10px 20px;
  }
}
</style>
