<template>
  <div class="list">
    <mpanel title="Experiments">
      <!-- 新建 -->
      <el-button :style="{marginBottom: '20px'}" type="primary" size="mini" @click="onAdd">New Experiment</el-button>
      <!-- 列表 -->
      <el-table v-loading="loading" :data="tableData" stripe style="width: 100%">
        <el-table-column prop="name" label="experiment name" align="center"/>
        <el-table-column label="training status" align="center">
          <template slot-scope="scope">
            <div class="flex-wrapper">
              <div :style="{background: status(scope.row.status).color}" :class="{ dot: true, breathe: scope.row.status === 3 }" ></div>
              {{ status(scope.row.status).text }}
            </div>
          </template>
        </el-table-column>
        <el-table-column label="total training time" align="center">
          <template slot-scope="scope">
            {{ scope.row.totalTrainingTime }} hours
          </template>
        </el-table-column>
        <el-table-column label="created at" align="center">
          <template slot-scope="scope">
            {{ moment(scope.row.createdAt).fromNow() }}
          </template>
        </el-table-column>
        <el-table-column label="operations" align="center" width="300">
          <template slot-scope="scope">
            <template v-if="scope.row.status === 1">
            </template>
            <template v-else-if="scope.row.status === 2">
              <el-button :style="{color: status(3).color}" type="text" size="mini"
                         @click="startTrainingProcess(scope.row.id)">start</el-button>
            </template>
            <template v-else>
              <el-button :style="{color: status(2).color}" type="text" size="mini"
                         @click="stopTrainingProcess(scope.row.id)">halt</el-button>
            </template>
            <el-button type="text" size="mini">
              <router-link :to="{ name: 'EditExperimentComponent', params: { experimentId: scope.row.id, componentName: 'dataset' }}">
                config
              </router-link>
            </el-button>
            <el-button :style="{color: '#F56C6C'}" type="text" size="mini" @click="delExperiment(scope.row.id)">delete</el-button>
          </template>
        </el-table-column>
      </el-table>
    </mpanel>
    <!-- 新建 -->
    <el-dialog :visible.sync="addVisible" :close-on-click-modal="false" class="add-form" title="New Experiment" width="500px" @closed="addCancel">
      <el-form :model="addForm">
        <el-form-item label="Experiment Name" required>
          <el-input v-model="addForm.name" placeholder="Input the experiment name, should be unique from others" autocomplete="off"/>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="addCancel">Cancel</el-button>
        <el-button :loading="addForm.loading" type="primary" @click="createExperiment">
          {{ addForm.loading ? 'Creating experiment' : 'Create experiment' }}
        </el-button>
      </div>
    </el-dialog>
  </div>
</template>
<script>
import mpanel from 'components/_common/mpanel'
import pagination from 'components/_common/pagination'
import dateFormat from 'dateformat'
import moment from 'moment'

export default {
  components: {
    mpanel,
    pagination
  },
  data () {
    return {
      addVisible: false,
      loading: false,
      categoryList: [],
      addForm: {
        name: '',
        loading: false
      },
      searchForm: {
        title: '',
        time: '',
        status: ''
      },
      statusText: {
        0: {
          text: '未运行'
        },
        1: {
          text: '运行中'
        }
      },
      tableData: [],
      currentPage: 1,
      pageSize: 10,
      total: 1
    }
  },
  mounted () {
    this.getData()
  },
  methods: {
    dateFormat,
    moment,
    // 获取列表
    async getData () {
      this.loading = true
      const experiments = await this.ajax.get('/api/experiments')
      this.tableData = experiments
      this.loading = false
    },
    status(status) {
      return {
        1: {
          color: '#F56C6C',
          text: 'stopped'
        },
        2: {
          color: '#E6A23C',
          text: 'idle'
        },
        3: {
          color: '#67C23A',
          text: 'running'
        }
      }[status]
    },
    // 搜索
    onSearch () {
      this.currentPage = 1
      this.getData()
    },
    // 新建
    onAdd () {
      this.addVisible = true
    },
    addCancel () {
      this.addVisible = false
      this.addForm = {
        titleSc: '',
        titleTc: '',
        titleEn: '',
        category: '',
        loading: false
      }
    },
    async createExperiment () {
      this.addForm.loading = true
      const response = await this.ajax.post('/api/experiments', {
        name: this.addForm.name
      })
      this.addForm.loading = false
      if (response) {
        this.getData()
        this.addCancel()
      }
    },
    async delExperiment(id) {
      await this.ajax.delete(`/api/experiments/${id}`)
      this.getData()
    },
    async startTrainingProcess(id) {
      await this.ajax.post(`/api/experiments/${id}/training`)
      this.getData()
      this.$notify({
        title: 'Start',
        message: 'The training process havs been started!',
        type: 'success'
      })
    },
    async stopTrainingProcess(id) {
      await this.ajax.post(`/api/experiments/${id}/halt`)
      this.getData()
      this.$notify({
        title: 'Halt',
        message: 'The training process havs been halted!',
        type: 'warning'
      })
    },
    // 翻页
    changePage (page, size) {
      this.currentPage = page
      this.pageSize = size
      this.getData()
    }
  }
}
</script>
<style lang="scss">
@keyframes breathe {
  0% {
    opacity: .2;
  }

  100% {
    opacity: 1;
  }
}

.searchpanel {
  padding: 20px;
  background-color: #fafafc;
  border: 1px solid #e5e5e5;
  margin-bottom: 20px;
  .el-form-item {
    margin-bottom: 0;
  }
}
.add-form {
  .el-select {
    width: 100%;
  }
}
.dot {
  height: 12px;
  width: 12px;
  margin-right: 10px;
  border-radius: 50%;
}
.flex-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
}

.breathe {
  animation: 1s linear 0s infinite alternate breathe;
}
</style>
