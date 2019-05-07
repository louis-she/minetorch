<template>
  <div class="list">
    <mpanel title="Experiments">
      <!-- 新建 -->
      <el-button
        :style="{marginBottom: '20px'}"
        type="primary"
        size="mini"
        @click="onAdd">New Experiment</el-button>
      <!-- 列表 -->
      <el-table
        v-loading="loading"
        :data="tableData"
        stripe
        style="width: 100%">
        <el-table-column
          prop="name"
          label="experiment name"
          align="center"/>
        <el-table-column
          label="training status"
          align="center">
          <template slot-scope="scope">
            {{ scope.row.isTraining ? 'training' : 'halt' }}
          </template>
        </el-table-column>
        <el-table-column
          label="total training time"
          align="center">
          <template slot-scope="scope">
            {{ scope.row.totalTrainingTime }} hours
          </template>
        </el-table-column>
        <el-table-column
          label="created at"
          align="center">
          <template slot-scope="scope">
            {{ moment(scope.row.createdAt).fromNow() }}
          </template>
        </el-table-column>
        <el-table-column
          label="operations"
          align="center"
          width="300">
          <template slot-scope="scope">
            <el-button
              :style="{color: '#67C23A'}"
              type="text"
              size="mini">start</el-button>
            <el-button
              type="text"
              size="mini">
              <router-link :to="{ name: 'EditExperimentDataset', params: { experimentId: scope.row.id }}">
                config
              </router-link>
            </el-button>
            <el-button
              :style="{color: '#F56C6C'}"
              type="text"
              size="mini"
              @click="delExperiment(scope.row.id)">delete</el-button>
          </template>
        </el-table-column>
      </el-table>
    </mpanel>
    <!-- 新建 -->
    <el-dialog
      :visible.sync="addVisible"
      :close-on-click-modal="false"
      class="add-form"
      title="New Experiment"
      width="500px"
      @closed="addCancel">
      <el-form :model="addForm">
        <el-form-item
          label="Experiment Name"
          required>
          <el-input
            v-model="addForm.name"
            placeholder="Input the experiment name, should be unique from others"
            autocomplete="off"/>
        </el-form-item>
      </el-form>
      <div
        slot="footer"
        class="dialog-footer">
        <el-button @click="addCancel">Cancel</el-button>
        <el-button
          :loading="addForm.loading"
          type="primary"
          @click="createExperiment">{{ addForm.loading ? 'Creating experiment' : 'Create experiment' }}</el-button>
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
</style>
